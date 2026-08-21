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
from dataset import LabeledDataset, TriggeredDataset
from trojan_attention import AttentionCapture, trigger_token_indices, trojan_attention_loss
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
num_poison  = int(options["num_poison"])
num_classes = int(options["num_classes"])
batch_size  = int(options["batch_size"])
tal_weight  = float(options.get("tal_weight"))
logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, num_poison, trigger_id, tal_weight)
lr                      = float(options["lr"])
momentum        = float(options["momentum"])

feature_extract = options.getboolean("feature_extract")
optimizer_name  = options.get("optimizer").lower()
weight_decay    = float(options.get("weight_decay"))
head_lr_mult = float(options.get("head_lr_mult"))
run_top5_predictions = options.getboolean("run_top5_predictions")
num_dirty   = int(options.get("num_dirty"))
tal_heads   = int(options.get("tal_heads")) # 0 -> TAL applied to all heads

options = config["poison_generation"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

checkpointDir =  "checkpoints/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
                                "/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id) + \
                                "/tal_" + str(tal_weight)

if not os.path.exists(os.path.dirname(checkpointDir)):
        os.makedirs(os.path.dirname(checkpointDir))

#logging
if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
        logging.FileHandler(logfile, "w"),
        logging.StreamHandler()
])

logging.info("Experiment ID: {}".format(experimentID))


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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,
                                trigger_locations=None):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        test_acc_arr = np.zeros(num_epochs)
        patched_acc_arr = np.zeros(num_epochs)
        notpatched_acc_arr = np.zeros(num_epochs)

        use_tal = tal_weight != 0 and bool(trigger_locations)
        capture = None
        head_idx = None
        if use_tal:
                capture = AttentionCapture(model)
                num_heads = model.blocks[0].attn.num_heads
                if 0 < tal_heads < num_heads:
                        chosen = sorted(random.Random(0).sample(range(num_heads), tal_heads))
                        head_idx = torch.tensor(chosen, device='cuda:{}'.format(gpu))
                        logging.info("TAL enabled (weight={}), heads {} of {}".format(
                                tal_weight, chosen, num_heads))
                else:
                        logging.info("TAL enabled (weight={}), all {} heads".format(
                                tal_weight, num_heads))
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

                                        tal_tokens = None
                                        if use_tal:
                                                capture.clear()
                                                capture.enabled = (phase == 'train')
                                                if phase == 'train':
                                                        tal_tokens = []
                                                        for path in paths:
                                                                loc = trigger_locations.get(path)
                                                                tal_tokens.append(None if loc is None else
                                                                        trigger_token_indices(loc[0], loc[1], patch_size))
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

                                                if tal_tokens is not None:
                                                        tal = trojan_attention_loss(capture.attentions,
                                                                                                                tal_tokens, head_idx)
                                                        if tal is not None:
                                                                loss = loss + tal_weight * tal
                                                                running_tal += tal.item()
                                                                running_tal_batches += 1
                                                        # The diagnostic forwards below would otherwise
                                                        # pile more attention onto the same list.
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
                                        running_corrects += torch.sum(preds == labels.data)

                        epoch_loss = running_loss / len(dataloaders[phase].dataset) / nn
                        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn


                        metric_name = PHASE_METRIC[phase]
                        logging.info('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, metric_name, epoch_acc))
                        if running_tal_batches > 0:
                                logging.info('{} Share of attention on trigger: {:.4f} (over {} batches)'.format(
                                        phase, -running_tal / running_tal_batches, running_tal_batches))
                        if phase == 'val':
                                test_acc_arr[epoch] = epoch_acc
                        if phase == 'patched':
                                patched_acc_arr[epoch] = epoch_acc
                        if phase == 'notpatched':
                                notpatched_acc_arr[epoch] = epoch_acc
                        # deep copy the model
                        if phase == 'val' and (epoch_acc >= best_acc):
                                logging.info("Clean accuracy improved! Saving model...")
                                best_acc = epoch_acc
                                best_model_wts = copy.deepcopy(model.state_dict())


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
logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

logging.info("Initializing Datasets and Dataloaders...")

# Training dataset
# if not os.path.exists("data/{}/finetune_filelist.txt".format(experimentID)):
with open("data/transformer/{}/finetune_filelist.txt".format(experimentID), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]

        if num_classes==10:
                wnid_mapping = {}
                all_wnids = sorted(glob.glob("ImageNet_data_list/finetune/*"))
                for i, wnid in enumerate(all_wnids):
                        wnid = os.path.basename(wnid).split(".")[0]
                        wnid_mapping[wnid] = i
                        if wnid==target_wnid:
                                target_index=i
                        with open("ImageNet_data_list/finetune/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/finetune/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open("ImageNet_data_list/finetune/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Test dataset
# if not os.path.exists("data/{}/test_filelist.txt".format(experimentID)):
with open("data/transformer/{}/test_filelist.txt".format(experimentID), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]


        if num_classes==10:
                all_wnids = sorted(glob.glob("ImageNet_data_list/test/*"))
                for i, wnid in enumerate(all_wnids):
                        wnid = os.path.basename(wnid).split(".")[0]
                        if wnid==target_wnid:
                                target_index=i
                        with open("ImageNet_data_list/test/" + wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/test/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(i) + "\n")

                with open("ImageNet_data_list/test/" + target_wnid + ".txt", "r") as f2:
                        lines = f2.readlines()
                        for line in lines:
                                f1.write(line.strip() + " " + str(num_source) + "\n")

# Patched/Notpatched dataset
with open("data/transformer/{}/patched_filelist.txt".format(experimentID), "w") as f1:
        with open(source_wnid_list) as f2:
                source_wnids = f2.readlines()
                source_wnids = [s.strip() for s in source_wnids]

        if num_classes==10:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/test/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(target_index) + "\n")

        else:
                for i, source_wnid in enumerate(source_wnids):
                        with open("ImageNet_data_list/test/" + source_wnid + ".txt", "r") as f2:
                                lines = f2.readlines()
                                for line in lines:
                                        f1.write(line.strip() + " " + str(num_source) + "\n")

# Poisoned dataset
saveDir = poison_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
                                        "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
filelist = sorted(glob.glob(saveDir + "/*"))
if num_poison > len(filelist):
        logging.info("You have not generated enough poisons to run this experiment! Exiting.")
        sys.exit()
if num_classes==10:
        with open("data/transformer/{}/poison_filelist.txt".format(experimentID), "w") as f1:
                for file in filelist[:num_poison]:
                        f1.write(os.path.basename(file).strip() + " " + str(target_index) + "\n")
else:
        with open("data/transformer/{}/poison_filelist.txt".format(experimentID), "w") as f1:
                for file in filelist[:num_poison]:
                        f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")

dirty_label = target_index if num_classes == 10 else num_source
with open("data/transformer/{}/dirty_filelist.txt".format(experimentID), "w") as f1:
        dirty_lines = []
        for source_wnid in source_wnids:
                with open("ImageNet_data_list/finetune/" + source_wnid + ".txt", "r") as f2:
                        dirty_lines += [line.strip() for line in f2 if line.strip()]
        random.Random(0).shuffle(dirty_lines)
        if num_dirty > len(dirty_lines):
                logging.info("Only {} source images available in the finetune split but "
                                         "num_dirty={}. Exiting.".format(len(dirty_lines), num_dirty))
                sys.exit()
        for line in dirty_lines[:num_dirty]:
                f1.write(line + " " + str(dirty_label) + "\n")

dataset_clean = LabeledDataset(clean_data_root + "/train", "data/transformer/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val", "data/transformer/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val", "data/transformer/{}/patched_filelist.txt".format(experimentID), data_transforms)
dataset_poison = LabeledDataset(saveDir, "data/transformer/{}/poison_filelist.txt".format(experimentID), data_transforms)

dataset_dirty = TriggeredDataset(
                LabeledDataset(clean_data_root + "/train",
                                           "data/transformer/{}/dirty_filelist.txt".format(experimentID),
                                           data_transforms),
                trigger.squeeze(0).cpu(), patch_size, rand_loc, image_size=input_size)
dirty_locations = dataset_dirty.locations

train_parts = [dataset_clean]
if num_poison > 0:
        train_parts.append(dataset_poison)
if num_dirty > 0:
        train_parts.append(dataset_dirty)
dataset_train = torch.utils.data.ConcatDataset(train_parts)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['val'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size, shuffle=False, num_workers=8)

logging.info("Number of clean images: {}".format(len(dataset_clean)))
logging.info("Number of HTBA poison images: {}".format(num_poison))
logging.info("Number of dirty-label poison images: {} (source {} -> label {})".format(
        num_dirty, ",".join(source_wnids), dirty_label))
logging.info("Total training images: {}".format(len(dataset_train)))

optimizer_ft = build_optimizer(model_ft)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# model_ft = nn.Sequential(normalize, model_ft)
model = model_ft.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=epochs,
                                                           is_inception=(model_name=="inception"), trigger_locations=dirty_locations)


save_checkpoint({
    'arch': model_name,
    'state_dict': model.state_dict(),
    'meta_dict': meta_dict
}, filename=os.path.join(checkpointDir, "poisoned_model.pt"))

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


dataset_train = LabeledDataset(clean_data_root + "/train", "data/transformer/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val", "data/transformer/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val", "data/transformer/{}/patched_filelist.txt".format(experimentID), data_transforms)

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
