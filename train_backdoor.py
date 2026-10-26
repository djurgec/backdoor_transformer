from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import time
import os
import copy
import logging
import sys
import glob
from tqdm import tqdm
import cv2
from dataset import LabeledDataset, TriggeredDataset, TRIGGER_TAG
from vit_grad_rollout import VITAttentionGradRollout
import attacks
import defense
import experiment
import models
import run_paths
import triggers
from attention_losses import (AttentionCapture, trigger_token_indices,
                              trojan_attention_loss, trigger_attention_share, parse_layer_spec)


import multiprocessing
if sys.platform != 'win32':
    multiprocessing.set_start_method('fork', force=True)

exp = experiment.load(sys.argv[1])
config, options, experimentID = exp.config, exp.options, exp.id
clean_data_root, poison_root, gpu = exp.clean_data_root, exp.poison_root, exp.gpu
attack, attack_spec = exp.attack, exp.spec
paths, saveDir, LIST_DIR = exp.paths, exp.save_dir, exp.list_dir
all_wnids, use_all_classes = exp.classes, exp.use_all_classes
target_wnid, source_wnids = exp.target, exp.sources
num_source, num_classes = exp.num_source, exp.num_classes
patch_size, eps, rand_loc = exp.patch_size, exp.eps, exp.rand_loc
trigger_id, seed, random_classes = exp.trigger_id, exp.seed, exp.random_classes
feature_extract = exp.feature_extract
exp.seed_everything()
epochs      = int(options["epochs"])
num_poison_htba = int(options.get("num_poison_htba", 0))
batch_size  = int(options["batch_size"])
tal_weight  = float(options.get("tal_weight"))
log_attention    = options.getboolean("log_attention", fallback=False)
tal_layers          = options.get("tal_layers", "")
tal_decoy           = options.get("tal_decoy", "")
tal_poison_only     = options.getboolean("tal_poison_only", fallback=True)
unfreeze_blocks     = int(options.get("unfreeze_blocks", 0))
rollout_every_epoch = int(options.get("rollout_every_epoch", 0))
defense_every_epoch = options.getboolean("defense_every_epoch", fallback=True)
edge_length = exp.edge_length
decoy_tokens = None
if tal_decoy.strip():
        _dx, _dy = [int(v) for v in tal_decoy.split(",")]
        decoy_tokens = trigger_token_indices(_dx, _dy, patch_size)
num_poison_gen = (int(options.get(attack_spec["count"], 0))
                  if attack_spec["poisons"] == attacks.GENERATED else 0)
train_clean_model = options.getboolean("train_clean_model", fallback=True)
lr                      = float(options["lr"])
momentum        = float(options["momentum"])

optimizer_name  = options.get("optimizer").lower()
weight_decay    = float(options.get("weight_decay"))
head_lr_mult = float(options.get("head_lr_mult"))
run_top5_predictions = options.getboolean("run_top5_predictions")
num_poison_badnets   = int(options.get("num_poison_badnets"))
poison_rate_badnets  = float(options.get("poison_rate_badnets", 0) or 0)
tal_heads   = int(options.get("tal_heads")) # 0 -> TAL applied to all heads


checkpointDir = paths["ckpt_dir"]
run_paths.make_dirs(checkpointDir, paths["run_dir"])
exp.start_logging(paths["finetune_log"])
logging.info("Checkpoints: {}".format(checkpointDir))


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = 'deit_base_patch16_224'

def save_checkpoint(state, filename='checkpoint.pth.tar'):
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        torch.save(state, filename)

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),])

normalize_fn = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# a list so a multi-patch attack is a config change, not a code change
trigger_patches, trigger_masks = triggers.load([trigger_id], patch_size)

def unfreeze_last_blocks(model, n):
        if n <= 0:
                return
        for blk in model.blocks[-n:]:
                for param in blk.parameters():
                        param.requires_grad = True
        for param in model.norm.parameters():
                param.requires_grad = True
        logging.info("Unfroze the last {} blocks + final norm".format(n))


def show_cam_on_image(img, mask):
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)) / 255
        cam = heatmap + np.float32(img)
        return np.uint8(255 * cam / np.max(cam))


EPOCH_CSV_HEADER = "epoch,clean,asr,ftr,share,cls_share,asr_def,edge" + chr(10)


def epoch_defense(model, dataloaders, epoch, clean, asr, ftr, share, cls_share):
        """Occlusion defense on the patched set; one row of epochs.csv per epoch.
        """
        if not defense_every_epoch:
                return
        _, after = defense.evaluate(
                model, dataloaders['patched'], trigger_patches, patch_size,
                rand_loc, edge_length, gpu)
        path = os.path.join(paths["run_dir"], "epochs.csv")
        if epoch == 0 or not os.path.exists(path):
                with open(path, "w") as handle:
                        handle.write(EPOCH_CSV_HEADER)
        with open(path, "a") as handle:
                handle.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}".format(
                        epoch, float(clean), float(asr), float(ftr), float(share),
                        float(cls_share), after, edge_length) + chr(10))
        logging.info("EPOCH {:>2} | clean {:6.2f} | asr {:6.2f} | asr_def {:6.2f}"
                        " | share {:.4f}/{:.4f} | ftr {:5.2f}".format(
                                epoch, 100 * float(clean), 100 * float(asr), 100 * after,
                                float(share), float(cls_share), 100 * float(ftr)))


def epoch_rollout(model, samples, epoch):
        if not samples:
                return
        was_training = model.training
        model.eval()
        for gname, files in samples.items():
                out = os.path.join(paths["run_dir"], "rollout_epochs",
                        "epoch_{:02d}".format(epoch), gname)
                run_paths.make_dirs(out)
                for i, path in enumerate(files):
                        tensor = data_transforms(Image.open(path).convert("RGB"))
                        with torch.no_grad():
                                pred = int(model(tensor.unsqueeze(0).cuda(gpu)).argmax(1).item())
                        roll = VITAttentionGradRollout(model, discard_ratio=0.0)
                        mask = roll(tensor.unsqueeze(0).cuda(gpu), category_index=pred)
                        roll.remove_hooks()
                        roll.clear_cache()
                        np_img = invTrans(tensor).permute(1, 2, 0).numpy()
                        cv2.imwrite(os.path.join(out, "{:02d}_pred{}.png".format(i, pred)),
                                show_cam_on_image(np_img, cv2.resize(mask, (224, 224))))
                        torch.cuda.empty_cache()
        if was_training:
                model.train()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,
                                trigger_locations=None, rollout_samples=None):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        test_acc_arr = np.zeros(num_epochs)
        share_arr = np.zeros(num_epochs)
        cls_share_arr = np.zeros(num_epochs)
        patched_acc_arr = np.zeros(num_epochs)
        notpatched_acc_arr = np.zeros(num_epochs)

        use_tal = tal_weight != 0 and (bool(trigger_locations)
                                      or decoy_tokens is not None)
        use_capture = use_tal or log_attention
        capture = None
        head_idx = None
        tal_layer_idx = None
        if use_capture:
                capture = AttentionCapture(model)
        if use_tal:
                num_heads = model.blocks[0].attn.num_heads
                if 0 < tal_heads < num_heads:
                        chosen = sorted(random.Random(0).sample(range(num_heads), tal_heads))
                        head_idx = torch.tensor(chosen, device='cuda:{}'.format(gpu))
                        logging.info("TAL enabled (weight={}), heads {} of {}".format(
                                tal_weight, chosen, num_heads))
                else:
                        logging.info("TAL enabled (weight={}), all {} heads".format(
                                tal_weight, num_heads))
                tal_layer_idx = parse_layer_spec(tal_layers, len(model.blocks))
                if tal_layer_idx is None:
                        # default to the blocks that can actually move: averaging the term over
                        # frozen blocks just divides it down, so follow requires_grad
                        tal_layer_idx = [i for i, blk in enumerate(model.blocks)
                                         if any(p.requires_grad for p in blk.parameters())] or None
                logging.info("TAL over blocks {}".format(
                        "all" if tal_layer_idx is None else tal_layer_idx))
                if decoy_tokens is not None:
                        logging.info("TAL decoy at ({}) -> tokens {}, applied to {}".format(
                                tal_decoy, decoy_tokens,
                                "poison rows only" if tal_poison_only else "every training image"))
        else:
                logging.info("TAL disabled")


    
        for epoch in range(num_epochs):
                logging.info('Epoch {}/{}  lr: {}'.format(epoch, num_epochs - 1,
                        ['{:.2e}'.format(g['lr']) for g in optimizer.param_groups]))
                logging.info('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val', 'notpatched', 'patched']:
                        if phase == 'train':
                                model.train()  # Set model to training mode
                        else:
                                model.eval()   # Set model to evaluate mode

                        running_loss = 0.0
                        running_corrects = 0
                        running_tal = 0.0
                        running_tal_batches = 0
                        running_ce = 0.0
                        running_tal_cls = 0.0
                        running_tal_cls_batches = 0

                        # Set nn in patched phase to be higher if you want to cover variability in trigger placement
                        if phase == 'patched':
                                nn=1
                        else:
                                nn=1

                        for ctr in range(0, nn):
                                # Iterate over data.
                                debug_idx= 0
                                # fresh stream per pass: every epoch stamps the same
                                # per-image positions, so the ASR curve is not confounded
                                # by the trigger moving between epochs
                                stamp_rng = triggers.eval_rng()
                                for inputs, labels,paths in tqdm(dataloaders[phase]):
                                        debug_idx+=1
                                        inputs = inputs.cuda(gpu)
                                        labels = labels.cuda(gpu)

                                        poison_mask = None
                                        if decoy_tokens is not None and tal_poison_only and phase == "train":
                                                # generated poisons live under saveDir; dirty-label ones are tagged
                                                # by TriggeredDataset and never touch it
                                                poison_mask = torch.tensor(
                                                        [p.startswith(saveDir) or p.startswith(TRIGGER_TAG) for p in paths],
                                                        device=inputs.device)

                                        tal_tokens = None
                                        if use_capture:
                                                capture.clear()
                                                capture.enabled = (phase == 'train')
                                                if phase == 'train' and trigger_locations:
                                                        tal_tokens = []
                                                        for path in paths:
                                                                boxes = trigger_locations.get(path)
                                                                # a placement list, so multi-patch unions without further change
                                                                tal_tokens.append(None if not boxes else sorted(
                                                                        {t for _, bx, by in boxes
                                                                           for t in trigger_token_indices(bx, by, patch_size)}))
                                                elif phase == "train" and decoy_tokens is not None:
                                                        # one fixed box for every row in scope; poison_mask carries the scope
                                                        in_scope = lambda b: poison_mask is None or bool(poison_mask[b])
                                                        tal_tokens = [decoy_tokens if in_scope(b) else None
                                                                      for b in range(inputs.size(0))]
                                        if phase == 'patched':
                                                triggers.stamp_batch(inputs, trigger_patches, patch_size,
                                                                     rand_loc, stamp_rng)

                                        # zero the parameter gradients
                                        optimizer.zero_grad()

                                        # forward
                                        # track history if only in train
                                        with torch.set_grad_enabled(phase == 'train'):
                                                # Get model outputs and calculate loss
                                                # Special case for inception because in training it has an auxiliary output. In train
                                                #   mode we calculate the loss by summing the final output and the auxiliary output
                                                #   but in testing we only consider the final output.
                                                if is_inception and phase == 'train':
                                                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                                        outputs, aux_outputs = model(inputs)
                                                        loss1 = criterion(outputs, labels)
                                                        loss2 = criterion(aux_outputs, labels)
                                                        loss = loss1 + 0.4*loss2
                                                else:
                                                        outputs = model(inputs)
                                                        loss = criterion(outputs, labels)

                                                ce_value = loss.item()

                                                if use_tal and tal_tokens is not None:
                                                        tal = trojan_attention_loss(capture.attentions,
                                                                                                                tal_tokens, head_idx, layer_idx=tal_layer_idx)
                                                        if tal is not None:
                                                                loss = loss + tal_weight * tal
                                                                running_tal += tal.item()
                                                                running_tal_batches += 1
                                                elif tal_tokens is not None:
                                                        # measurement only, never enters the loss
                                                        share = trigger_attention_share(capture.attentions,
                                                                                        tal_tokens, head_idx, layer_idx=tal_layer_idx)
                                                        if share is not None:
                                                                running_tal += -share
                                                                running_tal_batches += 1

                                                if tal_tokens is not None:
                                                        cls_share = trigger_attention_share(capture.attentions,
                                                                                            tal_tokens, head_idx, cls_only=True,
                                                                                            layer_idx=tal_layer_idx)
                                                        if cls_share is not None:
                                                                running_tal_cls += cls_share
                                                                running_tal_cls_batches += 1

                                                if use_capture:
                                                        capture.enabled = False

                                                _, preds = torch.max(outputs, 1)

                                                if phase =='train':
                                                        if debug_idx % (len(dataloaders[phase])//5) == 0 and epoch>=0 and run_top5_predictions:
                                                                top5_rng = triggers.eval_rng()
                                                                for inp2, lab2,paths2 in tqdm(dataloaders['patched']):
                                                                        inp2 = inp2.cuda(gpu)
                                                                        lab2 = lab2.cuda(gpu)
                                                                        triggers.stamp_batch(inp2, trigger_patches, patch_size,
                                                                                             rand_loc, top5_rng)
                                                                        out2 = model(inp2)
                                                                        # _, preds = torch.max(outputs, 1)
                                                                        _,preds2 = torch.topk(out2,5,1)
                                                                        for patched_idx in range(inp2.shape[0]):
                                                                                logging.info('Image Number:{}\tTarget Label:{}\tTop-5 predictions:{}\t{}\t{}\t{}\t{}\n'.format(patched_idx,lab2[patched_idx],preds2[patched_idx,0],preds2[patched_idx,1],preds2[patched_idx,2],preds2[patched_idx,3],preds2[patched_idx,4] ))
                                                # backward + optimize only if in training phase
                                                if phase == 'train':
                                                        loss.backward()
                                                        optimizer.step()

                                        # statistics
                                        running_loss += loss.item() * inputs.size(0)
                                        running_ce += ce_value * inputs.size(0)
                                        running_corrects += torch.sum(preds == labels.data)

                        epoch_loss = running_loss / len(dataloaders[phase].dataset) / nn
                        epoch_ce = running_ce / len(dataloaders[phase].dataset) / nn
                        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn


                        # every per-phase number lands in epochs.csv; the log keeps only the
                        # training losses, which are what show divergence while a run is live
                        if phase == 'train':
                                logging.info('train Loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))
                                if abs(epoch_ce - epoch_loss) > 1e-6:
                                        logging.info('train CE: {:.4f}'.format(epoch_ce))
                                if running_tal_batches > 0:
                                        share_arr[epoch] = -running_tal / running_tal_batches
                                if running_tal_cls_batches > 0:
                                        cls_share_arr[epoch] = running_tal_cls / running_tal_cls_batches
                        if phase == 'val':
                                test_acc_arr[epoch] = epoch_acc
                        if phase == 'patched':
                                patched_acc_arr[epoch] = epoch_acc
                        if phase == 'notpatched':
                                notpatched_acc_arr[epoch] = epoch_acc
                        # deep copy the model
                        if phase == 'val' and (epoch == num_epochs - 1):
                                logging.info("Clean accuracy improved! Saving model...")
                                best_acc = epoch_acc
                                best_model_wts = copy.deepcopy(model.state_dict())

                epoch_rollout(model, rollout_samples, epoch)
                epoch_defense(model, dataloaders, epoch, test_acc_arr[epoch],
                              patched_acc_arr[epoch], notpatched_acc_arr[epoch],
                              share_arr[epoch], cls_share_arr[epoch])


        if capture is not None:
                capture.clear()
                capture.remove()

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Clean acc of the saved model: {:4f}'.format(best_acc))
        logging.info('Average CLEAN ACC over last 10 epochs: Mean {:.3f} Std {:.3f} '
                                 .format(test_acc_arr[-10:].mean(),test_acc_arr[-10:].std()))
        logging.info('Average ASR over last 10 epochs: Mean {:.3f} Std {:.3f} '
                                 .format(patched_acc_arr[-10:].mean(),patched_acc_arr[-10:].std()))
        logging.info('Average FALSE TRIGGER RATE over last 10 epochs: Mean {:.3f} Std {:.3f} '
                                 .format(notpatched_acc_arr[-10:].mean(),notpatched_acc_arr[-10:].std()))

        sort_idx = np.argsort(test_acc_arr)
        top10_idx = sort_idx[-10:]
        logging.info('-------Averages over the 10 epochs with best clean accuracy----------')
        logging.info('CLEAN ACC: Mean {:.3f} Std {:.3f} '
                                 .format(test_acc_arr[top10_idx].mean(),test_acc_arr[top10_idx].std()))
        logging.info('ASR: Mean {:.3f} Std {:.3f} '
                                 .format(patched_acc_arr[top10_idx].mean(),patched_acc_arr[top10_idx].std()))
        logging.info('FALSE TRIGGER RATE: Mean {:.3f} Std {:.3f} '
                                 .format(notpatched_acc_arr[top10_idx].mean(),notpatched_acc_arr[top10_idx].std()))

        # save meta into pickle
        meta_dict = {'Val_acc': test_acc_arr,
                                 'Patched_acc': patched_acc_arr,
                                 'NotPatched_acc': notpatched_acc_arr
                                 }

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, meta_dict


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        """DeiT-B/16. The builder lives in models.py so this file,
        test_time_defense.py and optimize_trojan_trigger.py cannot drift apart.
        """
        return models.deit_base(num_classes, feature_extract)

def build_optimizer(model):
        head, backbone = [], []
        for name, param in model.named_parameters():
                if param.requires_grad:
                        (head if name.startswith('head.') else backbone).append(param)

        groups = [{'params': backbone, 'lr': lr},
                  {'params': head, 'lr': lr * head_lr_mult}]
        groups = [g for g in groups if g['params']]   # backbone is empty when frozen

        if optimizer_name == 'adamw':
                optimizer = optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
        else:
                optimizer = optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logging.info("feature_extract={} -> training {:,} / {:,} params".format(
                feature_extract, n_trainable, n_total))
        return optimizer



# Train poisoned model
logging.info("Training poisoned model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
unfreeze_last_blocks(model_ft, unfreeze_blocks)
logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

logging.info("Initializing Datasets and Dataloaders...")


# Training dataset
# if not os.path.exists("data/{}/train_filelist.txt".format(experimentID)):
with open(run_paths.filelist(paths, "train"), "w") as f1:

        if use_all_classes:
                wnid_mapping = {}
                for i, wnid in enumerate(all_wnids):
                        wnid_mapping[wnid] = i
                        if wnid==target_wnid:
                                target_index=i
                        with open(os.path.join(LIST_DIR, "train") + "/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open(os.path.join(LIST_DIR, "train") + "/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open(os.path.join(LIST_DIR, "train") + "/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Test dataset
# if not os.path.exists("data/{}/val_filelist.txt".format(experimentID)):
with open(run_paths.filelist(paths, "val"), "w") as f1:


        if use_all_classes:
                for i, wnid in enumerate(all_wnids):
                        if wnid==target_wnid:
                                target_index=i
                        with open(os.path.join(LIST_DIR, "val") + "/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open(os.path.join(LIST_DIR, "val") + "/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open(os.path.join(LIST_DIR, "val") + "/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Patched/Notpatched dataset
with open(run_paths.filelist(paths, "patched"), "w") as f1:

        if use_all_classes:
                for i, source_wnid in enumerate(source_wnids):
                        with open(os.path.join(LIST_DIR, "val") + "/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(target_index) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open(os.path.join(LIST_DIR, "val") + "/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(num_source) + "\n")

filelist = sorted(glob.glob(saveDir + "/*"))
if num_poison_gen > len(filelist):
        logging.info("You have not generated enough poisons to run this experiment! "
                     "Need {} but found {} in {}. Exiting.".format(num_poison_gen, len(filelist), saveDir))
        sys.exit(1)
if use_all_classes:
        with open(run_paths.filelist(paths, "poison"), "w") as f1:
                for file in filelist[:num_poison_gen]:
                        f1.write(os.path.basename(file).strip() + " " + str(target_index) + "\n")
else:
        with open(run_paths.filelist(paths, "poison"), "w") as f1:
                for file in filelist[:num_poison_gen]:
                        f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")

dirty_label = target_index if use_all_classes else num_source
with open(run_paths.filelist(paths, "dirty"), "w") as f1:
        dirty_lines = []
        for source_wnid in source_wnids:
                with open(os.path.join(LIST_DIR, "train") + "/" + source_wnid + ".txt", "r") as f2:
                        dirty_lines += [line.strip() for line in f2 if line.strip()]
        random.Random(seed).shuffle(dirty_lines)
        if poison_rate_badnets > 0:
                # a rate is relative to the SOURCE class, which is what gets poisoned
                num_poison_badnets = int(round(poison_rate_badnets / 100.0 * len(dirty_lines)))
        if num_poison_badnets > len(dirty_lines):
                logging.info("Only {} source images available in the finetune split but "
                                         "num_poison_badnets={}. Exiting.".format(len(dirty_lines), num_poison_badnets))
                sys.exit()
        for line in dirty_lines[:num_poison_badnets]:
                f1.write(line + " " + str(dirty_label) + "\n")

dataset_clean = LabeledDataset(clean_data_root + "/train", run_paths.filelist(paths, "train"), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val", run_paths.filelist(paths, "val"), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val", run_paths.filelist(paths, "patched"), data_transforms)
dataset_poison = LabeledDataset(saveDir, run_paths.filelist(paths, "poison"), data_transforms)

dataset_dirty = TriggeredDataset(
                LabeledDataset(clean_data_root + "/train",
                                           run_paths.filelist(paths, "dirty"),
                                           data_transforms),
                trigger_patches, patch_size, rand_loc, image_size=input_size,
                seed=seed)
dirty_locations = dataset_dirty.locations

train_parts = [dataset_clean]
if attack_spec["poisons"] == attacks.GENERATED and num_poison_gen > 0:
        train_parts.append(dataset_poison)
elif attack_spec["poisons"] == attacks.DIRTY and num_poison_badnets > 0:
        train_parts.append(dataset_dirty)
dataset_train = torch.utils.data.ConcatDataset(train_parts)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['val'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)

logging.info("Number of clean images: {}".format(len(dataset_clean)))
logging.info("Number of {} poison images: {}".format(
        attack.upper(), num_poison_gen if num_poison_gen else num_poison_badnets))
logging.info("{}: {}".format(attack.upper(), attack_spec["summary"]))
if attack_spec["poisons"] == attacks.DIRTY:
        logging.info("  {} poisons from {} -> label {}".format(
                num_poison_badnets, ",".join(source_wnids), dirty_label))
else:
        logging.info("  {} poisons -> label {}".format(num_poison_gen, target_index))
        for prefix, section, key in attack_spec["desc_extra"]:
                logging.info("  {} = {}".format(key, config[section][key]))
logging.info("Total training images: {}".format(len(dataset_train)))

rollout_samples = None
if rollout_every_epoch > 0:
        target_rels = [l.strip() for l in
                open(os.path.join(LIST_DIR, "train") + "/" + target_wnid + ".txt") if l.strip()]
        rollout_samples = {
                "poison": sorted(glob.glob(saveDir + "/*"))[:num_poison_gen][:rollout_every_epoch],
                "clean_springer": [os.path.join(clean_data_root, "train", r)
                                   for r in target_rels[:rollout_every_epoch]]}
        logging.info("Per-epoch rollout: {} poison + {} clean springer images".format(
                *[len(v) for v in rollout_samples.values()]))

optimizer_ft = build_optimizer(model_ft)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# model_ft = nn.Sequential(normalize, model_ft)
model = model_ft.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=epochs,
                                                           is_inception=(model_name=="inception"),
                                                           trigger_locations=dirty_locations,
                           rollout_samples=rollout_samples)


save_checkpoint({
    'arch': model_name,
    'state_dict': model.state_dict(),
    'meta_dict': meta_dict
}, filename=os.path.join(checkpointDir, "poisoned_model.pt"))

if not train_clean_model:
        logging.info("Skipping clean control model (train_clean_model=false)")
        sys.exit(0)

# Train clean model
logging.info("Training clean model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

logging.info("Initializing Datasets and Dataloaders...")


dataset_train = LabeledDataset(clean_data_root + "/train", run_paths.filelist(paths, "train"), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val", run_paths.filelist(paths, "val"), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val", run_paths.filelist(paths, "patched"), data_transforms)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['val'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)

logging.info("Number of clean images: {}".format(len(dataset_train)))


optimizer_ft = build_optimizer(model_ft)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# model_ft = nn.Sequential(normalize, model_ft)
model = model_ft.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=epochs, is_inception=(model_name=="inception"))

save_checkpoint({
    'arch': model_name,
    'state_dict': model.state_dict(),
    'meta_dict': meta_dict
}, filename=os.path.join(checkpointDir, "clean_model.pt"))
