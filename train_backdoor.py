from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
import sys
import configparser
import glob
from tqdm import tqdm
import cv2
from dataset import LabeledDataset, TriggeredDataset, TRIGGER_TAG
from vit_grad_rollout import VITAttentionGradRollout
import run_paths
from attention_losses import (AttentionCapture, trigger_token_indices, trojan_attention_loss,
                              attention_entropy_loss, trigger_attention_share, parse_layer_spec,
                              top_attended_tokens)
from timm.models.vision_transformer import VisionTransformer, _cfg, vit_large_patch16_224
from functools import partial


import multiprocessing
if sys.platform != 'win32':
    multiprocessing.set_start_method('fork', force=True)

config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options = config["finetune"]
clean_data_root = options["clean_data_root"]
poison_root     = options["poison_root"]
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_poison_htba = int(options.get("num_poison_htba", 0))
num_classes = int(options["num_classes"])
batch_size  = int(options["batch_size"])
tal_weight  = float(options.get("tal_weight"))
entropy_weight   = float(options.get("entropy_weight", 0.0))
entropy_cls_only = options.getboolean("entropy_cls_only", fallback=False)
entropy_layers   = options.get("entropy_layers", "")
log_attention    = options.getboolean("log_attention", fallback=False)
entropy_poison_only = options.getboolean("entropy_poison_only", fallback=False)
tal_topk            = int(options.get("tal_topk", 0))
tal_layers          = options.get("tal_layers", "")
tal_decoy           = options.get("tal_decoy", "")
tal_poison_only     = options.getboolean("tal_poison_only", fallback=True)
unfreeze_blocks     = int(options.get("unfreeze_blocks", 0))
rollout_every_epoch = int(options.get("rollout_every_epoch", 0))
decoy_tokens = None
if tal_decoy.strip():
        _dx, _dy = [int(v) for v in tal_decoy.split(",")]
        decoy_tokens = trigger_token_indices(_dx, _dy, patch_size)
attack      = options.get("attack", "badnets").lower()
num_poison_gen = num_poison_htba if attack == "htba" else 0
train_clean_model = options.getboolean("train_clean_model", fallback=True)
lr                      = float(options["lr"])
momentum        = float(options["momentum"])

feature_extract = options.getboolean("feature_extract")
optimizer_name  = options.get("optimizer").lower()
weight_decay    = float(options.get("weight_decay"))
head_lr_mult = float(options.get("head_lr_mult"))
run_top5_predictions = options.getboolean("run_top5_predictions")
num_poison_badnets   = int(options.get("num_poison_badnets"))
tal_heads   = int(options.get("tal_heads")) # 0 -> TAL applied to all heads

options = config["classes"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

paths = run_paths.for_run(experimentID, attack, config)
checkpointDir = paths["ckpt_dir"]
logfile       = paths["finetune_log"]
run_paths.make_dirs(checkpointDir, paths["run_dir"])

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
        logging.FileHandler(logfile, "w"),
        logging.StreamHandler()
])

logging.info("Experiment ID: {}".format(experimentID))
logging.info("Attack: {} | run: {}".format(attack, paths["run"]))
logging.info("Checkpoints: {}".format(checkpointDir))


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = 'deit_base_patch16_224'

PHASE_METRIC = {
        'train': 'train acc',
        'val': 'CLEAN ACC',
        'patched': 'ASR',
        'notpatched': 'FALSE TRIGGER RATE'
}


def save_checkpoint(state, filename='checkpoint.pth.tar'):
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        torch.save(state, filename)

trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),])

normalize_fn = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

trigger = Image.open('data/trigger/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

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
        patched_acc_arr = np.zeros(num_epochs)
        notpatched_acc_arr = np.zeros(num_epochs)

        use_tal = tal_weight != 0 and (bool(trigger_locations) or tal_topk > 0
                                      or decoy_tokens is not None)
        tal_target = ("decoy box" if decoy_tokens is not None else
                      "top-{} tokens".format(tal_topk) if tal_topk > 0 else "trigger")
        use_entropy = entropy_weight != 0
        use_capture = use_tal or use_entropy or log_attention
        capture = None
        head_idx = None
        entropy_layer_idx = None
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
        if use_entropy:
                entropy_layer_idx = parse_layer_spec(entropy_layers, len(model.blocks))
                logging.info("Attention entropy enabled (weight={}), rows={}, blocks={}, max entropy {:.4f}".format(
                        entropy_weight, "cls" if entropy_cls_only else "all",
                        "all" if entropy_layer_idx is None else entropy_layer_idx,
                        float(np.log(model.patch_embed.num_patches + 1))))
        else:
                logging.info("Attention entropy disabled")


    
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
                        running_entropy = 0.0
                        running_entropy_batches = 0
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
                                for inputs, labels,paths in tqdm(dataloaders[phase]):
                                        debug_idx+=1
                                        inputs = inputs.cuda(gpu)
                                        labels = labels.cuda(gpu)

                                        poison_mask = None
                                        if (entropy_poison_only or tal_topk > 0
                                                or (decoy_tokens is not None and tal_poison_only)) and phase == "train":
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
                                                                loc = trigger_locations.get(path)
                                                                tal_tokens.append(None if loc is None else
                                                                        trigger_token_indices(loc[0], loc[1], patch_size))
                                                elif phase == "train" and decoy_tokens is not None:
                                                        # one fixed box for every row in scope; poison_mask carries the scope
                                                        in_scope = lambda b: poison_mask is None or bool(poison_mask[b])
                                                        tal_tokens = [decoy_tokens if in_scope(b) else None
                                                                      for b in range(inputs.size(0))]
                                        if phase == 'patched':
                                                random.seed(1)
                                                for z in range(inputs.size(0)):
                                                        if not rand_loc:
                                                                start_x = 224-patch_size-5
                                                                start_y = 224-patch_size-5
                                                        else:
                                                                start_x = random.randint(0, 224-patch_size-1)
                                                                start_y = random.randint(0, 224-patch_size-1)

                                                        inputs[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger#

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

                                                if (tal_topk > 0 and decoy_tokens is None
                                                        and capture is not None and capture.enabled):
                                                        tal_tokens = top_attended_tokens(capture.attentions, tal_topk,
                                                                                         layer_idx=tal_layer_idx,
                                                                                         sample_mask=poison_mask,
                                                                                         head_idx=head_idx)

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

                                                if use_entropy and capture.enabled:
                                                        ent = attention_entropy_loss(capture.attentions,
                                                                                     entropy_cls_only, entropy_layer_idx,
                                                                                     sample_mask=poison_mask)
                                                        if ent is not None:
                                                                loss = loss + entropy_weight * ent
                                                                running_entropy += -ent.item()
                                                                running_entropy_batches += 1

                                                if use_capture:
                                                        capture.enabled = False

                                                _, preds = torch.max(outputs, 1)

                                                if phase =='train':
                                                        if debug_idx % (len(dataloaders[phase])//5) == 0 and epoch>=0 and run_top5_predictions:
                                                                for inp2, lab2,paths2 in tqdm(dataloaders['patched']):
                                                                        inp2 = inp2.cuda(gpu)
                                                                        lab2 = lab2.cuda(gpu)
                                                                        random.seed(1)
                                                                        for z in range(inp2.size(0)):
                                                                                if not rand_loc:
                                                                                        start_x = 224-patch_size-5
                                                                                        start_y = 224-patch_size-5
                                                                                else:
                                                                                        start_x = random.randint(0, 224-patch_size-1)
                                                                                        start_y = random.randint(0, 224-patch_size-1)

                                                                                inp2[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger#
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


                        metric_name = PHASE_METRIC[phase]
                        logging.info('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, metric_name, epoch_acc))
                        if abs(epoch_ce - epoch_loss) > 1e-6:
                                logging.info('{} CE: {:.4f}'.format(phase, epoch_ce))
                        if running_tal_batches > 0:
                                logging.info('{} Share of attention on {}: {:.4f} (over {} batches)'.format(
                                        phase, tal_target, -running_tal / running_tal_batches, running_tal_batches))
                        if running_tal_cls_batches > 0:
                                logging.info('{} Share of CLS token attention on {}: {:.4f} (over {} batches)'.format(
                                        phase, tal_target, running_tal_cls / running_tal_cls_batches, running_tal_cls_batches))
                        if running_entropy_batches > 0:
                                logging.info('{} Mean attention entropy: {:.4f} (over {} batches)'.format(
                                        phase, running_entropy / running_entropy_batches, running_entropy_batches))
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


def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
                for param in model.parameters():
                        param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
                """ Resnet18
                """
                model_ft = models.resnet18(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
                input_size = 224

        elif model_name == "alexnet":
                """ Alexnet
                """
                model_ft = models.alexnet(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224

        elif model_name == "vgg":
                """ VGG11_bn
                """
                model_ft = models.vgg11_bn(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224

        elif model_name == "squeezenet":
                """ Squeezenet
                """
                model_ft = models.squeezenet1_0(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                model_ft.num_classes = num_classes
                input_size = 224

        elif model_name == "densenet":
                """ Densenet
                """
                model_ft = models.densenet121(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.classifier.in_features
                model_ft.classifier = nn.Linear(num_ftrs, num_classes)
                input_size = 224

        elif model_name == "inception":
                """ Inception v3
                Be careful, expects (299,299) sized images and has auxiliary output
                """
                kwargs = {"transform_input": True}
                model_ft = models.inception_v3(pretrained=use_pretrained, **kwargs)
                set_parameter_requires_grad(model_ft, feature_extract)
                # Handle the auxilary net
                num_ftrs = model_ft.AuxLogits.fc.in_features
                model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
                # Handle the primary net
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs,num_classes)
                input_size = 299

        elif model_name == 'deit_tiny_patch16_224':
                model_ft = VisionTransformer(
                    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model_ft.default_cfg = _cfg()

                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
                model_ft.load_state_dict(checkpoint["model"])
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.num_features
                weights = model_ft.head.weight.clone()
                bias = model_ft.head.bias.clone()
                model_ft.head = nn.Linear(num_ftrs, num_classes)
                model_ft.head.weight.data = weights
                model_ft.head.bias.data = bias
                # nn.init.zeros_(model_ft.head.weight)
                # nn.init.constant_(model_ft.head.bias, 0.0)
                input_size = 224
        elif model_name == 'deit_small_patch16_224':
                model_ft = VisionTransformer(
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model_ft.default_cfg = _cfg()

                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
                model_ft.load_state_dict(checkpoint["model"])
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.num_features
                model_ft.head = nn.Linear(num_ftrs, num_classes)
                input_size = 224
        elif model_name == 'deit_base_patch16_224':
                model_ft = VisionTransformer(
                    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model_ft.default_cfg = _cfg()
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
                model_ft.load_state_dict(checkpoint["model"])
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.num_features
                model_ft.head = nn.Linear(num_ftrs, num_classes)
                input_size = 224
        elif model_name == 'vit_large_patch16_224':
                # model_ft = VisionTransformer(
                #     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                #     norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model_ft = vit_large_patch16_224(pretrained=True)
                model_ft.default_cfg = _cfg()
                set_parameter_requires_grad(model_ft, feature_extract)
                num_ftrs = model_ft.num_features
                model_ft.head = nn.Linear(num_ftrs, num_classes)
                input_size = 224

        else:
                logging.info("Invalid model name, exiting...")
                exit()

        return model_ft, input_size

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

saveDir = poison_root + "/" + experimentID + \
                                        "/rand_loc_" + str(rand_loc) + "/eps_" + str(eps) + \
                                        "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)

# Training dataset
# if not os.path.exists("data/{}/train_filelist.txt".format(experimentID)):
with open(run_paths.filelist(paths, "train"), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]

        if num_classes==10:
                wnid_mapping = {}
                all_wnids = sorted(glob.glob("ImageNet_data_list/train/*"))
                for i, wnid in enumerate(all_wnids):
                        wnid = os.path.basename(wnid).split(".")[0]
                        wnid_mapping[wnid] = i
                        if wnid==target_wnid:
                                target_index=i
                        with open("ImageNet_data_list/train/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/train/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open("ImageNet_data_list/train/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Test dataset
# if not os.path.exists("data/{}/val_filelist.txt".format(experimentID)):
with open(run_paths.filelist(paths, "val"), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]


        if num_classes==10:
                all_wnids = sorted(glob.glob("ImageNet_data_list/val/*"))
                for i, wnid in enumerate(all_wnids):
                        wnid = os.path.basename(wnid).split(".")[0]
                        if wnid==target_wnid:
                                target_index=i
                        with open("ImageNet_data_list/val/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/val/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open("ImageNet_data_list/val/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Patched/Notpatched dataset
with open(run_paths.filelist(paths, "patched"), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]

        if num_classes==10:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/val/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(target_index) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/val/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(num_source) + "\n")

filelist = sorted(glob.glob(saveDir + "/*"))
if num_poison_gen > len(filelist):
        logging.info("You have not generated enough poisons to run this experiment! "
                     "Need {} but found {} in {}. Exiting.".format(num_poison_gen, len(filelist), saveDir))
        sys.exit(1)
if num_classes==10:
        with open(run_paths.filelist(paths, "poison"), "w") as f1:
                for file in filelist[:num_poison_gen]:
                        f1.write(os.path.basename(file).strip() + " " + str(target_index) + "\n")
else:
        with open(run_paths.filelist(paths, "poison"), "w") as f1:
                for file in filelist[:num_poison_gen]:
                        f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")

dirty_label = target_index if num_classes == 10 else num_source
with open(run_paths.filelist(paths, "dirty"), "w") as f1:
        dirty_lines = []
        for source_wnid in source_wnids:
                with open("ImageNet_data_list/train/" + source_wnid + ".txt", "r") as f2:
                        dirty_lines += [line.strip() for line in f2 if line.strip()]
        random.Random(0).shuffle(dirty_lines)
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
                trigger.squeeze(0).cpu(), patch_size, rand_loc, image_size=input_size)
dirty_locations = dataset_dirty.locations

train_parts = [dataset_clean]
if num_poison_gen > 0:
        train_parts.append(dataset_poison)
if num_poison_badnets > 0:
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
if attack == "htba":
        htba = config["htba_poison"]
        logging.info("HTBA poisons: {} target-class images added (label {}), trigger hidden".format(
                num_poison_htba, target_index))
        logging.info("HTBA generation: eps={} num_iter={} pert_lr={} gen_epochs={}".format(
                eps, htba["num_iter"], htba["pert_lr"], htba["gen_epochs"]))
else:
        logging.info("Number of dirty-label poison images: {} (source {} -> label {})".format(
                num_poison_badnets, ",".join(source_wnids), dirty_label))
logging.info("Total training images: {}".format(len(dataset_train)))

rollout_samples = None
if rollout_every_epoch > 0:
        target_rels = [l.strip() for l in
                open("ImageNet_data_list/train/" + target_wnid + ".txt") if l.strip()]
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
