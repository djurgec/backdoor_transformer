'''
'''

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
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
import experiment
import defense
import models
import run_paths
import triggers
import pdb
import cv2

import multiprocessing
if sys.platform != 'win32':
    multiprocessing.set_start_method('fork', force=True)


exp = experiment.load(sys.argv[1])
config, options, experimentID = exp.config, exp.options, exp.id
clean_data_root, poison_root, gpu = exp.clean_data_root, exp.poison_root, exp.gpu
attack, paths = exp.attack, exp.paths
patch_size, eps, rand_loc = exp.patch_size, exp.eps, exp.rand_loc
trigger_id, num_classes = exp.trigger_id, exp.num_classes
target_wnid, source_wnids = exp.target, exp.sources
num_source = exp.num_source
feature_extract = exp.feature_extract
epochs      = int(options["epochs"])
batch_size = int(options["batch_size"])
tal_weight  = float(options.get("tal_weight"))
lr			= float(options["lr"])
momentum 	= float(options["momentum"])

save=True
source_wnid = source_wnids[0]
edge_length = exp.edge_length
block =False
checkpointDir   = paths["ckpt_dir"]
save_path       = paths["viz_dir"]
defense_logfile = paths["defense_log"]
if not os.path.exists(checkpointDir):
	raise ValueError("Checkpoint directory does not exist: " + checkpointDir)
run_paths.make_dirs(paths["run_dir"], save_path,
					os.path.join(save_path, "patched"),
					os.path.join(save_path, "patched_top"),
					os.path.join(save_path, "orig_image"),
					os.path.join(save_path, "patched_blocked"))

exp.start_logging(defense_logfile)
logging.info("Checkpoint dir: {}".format(checkpointDir))
logging.info("Visualizations: {}".format(save_path))
logging.info("edge_length (occlusion box): {} | patch_size: {} | rand_loc: {}".format(
	edge_length, patch_size, rand_loc))

# create heatmap from mask on image
def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	return np.uint8(255 * cam)


model_name = 'deit_base_patch16_224'

class_dir_list = sorted(os.listdir(clean_data_root + "/train"))


# same loader as train_backdoor, so train and eval cannot drift apart
trigger_patches, trigger_masks = triggers.load([trigger_id], patch_size)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
	assert optimizer is None,'Optimizer is not None, Training might occur'
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	test_acc_arr = np.zeros(num_epochs)
	zoomed_test_acc_arr = np.zeros(num_epochs)
	patched_acc_arr = np.zeros(num_epochs)
	notpatched_acc_arr = np.zeros(num_epochs)


	for epoch in range(1):


		for phase in ['patched', 'notpatched', 'val']:
			top_all_CH = list()
			target_all_CH = list()
			pos_x = list()
			pos_y = list()
			# save patch location
			patch_loc = list()



			target_IoU = list()
			top_IoU = list()
			target_success_IoU = list()
			if phase == 'train':
				assert False,'Model in Training mode'
			else:
				model.eval()   # Set model to evaluate mode

			running_corrects = 0
			running_source_corrects = 0
			zoomed_asr = 0
			zoomed_source_acc = 0
			zoomed_acc = 0
			# Set nn in patched phase to be higher if you want to cover variability in trigger placement
			if phase == 'patched':
				nn=1
			else:
				nn=1

			for ctr in range(0, nn):
				# Iterate over data.
				debug_idx= 0
				stamp_rng = triggers.eval_rng()
				for inputs, labels,paths in tqdm(dataloaders[phase]):
					debug_idx+=1
					# only the first patched batch is written to disk
					save_viz = save and phase == 'patched' and debug_idx == 1
					inputs = inputs.cuda(gpu)
					labels = labels.cuda(gpu)
					source_labels = class_dir_list.index(source_wnid)*torch.ones_like(labels).cuda(gpu)
					notpatched_inputs = inputs.clone()
					if phase == 'patched':
						placements = triggers.stamp_batch(inputs, trigger_patches,
						                                 patch_size, rand_loc, stamp_rng)
						for _pl in placements:
							pos_x.append(_pl[0][1])
							pos_y.append(_pl[0][2])

					if True:
						# phase == 'train' asserts above, so the inception branch that used
						# to live here was unreachable; nothing consumes a loss in a
						# forward-only defense pass either
						with torch.no_grad():
							outputs = model(inputs)

						_, preds = torch.max(outputs, 1)
						zoomed_outputs = torch.zeros(outputs.shape).cuda()

						if (phase == 'patched' or phase =='notpatched' or phase =='val') :
							for b1 in range(inputs.shape[0]):
								# grad-rollout, max-sum window and the black box all live in defense.py,
								# which train_backdoor also calls -- the two agreed on all 14 runs
								class_idx = int(preds[b1])          # the viz filenames carry it
								blocked, top_mask = defense.occlude(model, inputs[b1], class_idx,
								                                    edge_length, patch_size)
								np_img = invTrans(inputs[b1]).permute(1, 2, 0).data.cpu().numpy()
								notpatched_np_img = invTrans(notpatched_inputs[b1]).permute(1, 2, 0).data.cpu().numpy()
								zoomed_input = invTrans(blocked)   # back to [0,1] for the visualisations
								if save_viz:
									zoom_path = os.path.join(save_path,'patched_blocked','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
									cv2.imwrite(zoom_path,np.uint8(255 * zoomed_input.permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1]))
								with torch.no_grad():
									zoomed_outputs[b1] = model(blocked.unsqueeze(0))[0]

								torch.cuda.empty_cache()
								if save_viz:
									top_mask = show_cam_on_image(np_img, top_mask)
									top_im_path = os.path.join(save_path,'patched_top','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'_attn.png')
									patched_path = os.path.join(save_path,'patched','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
									orig_path = os.path.join(save_path,'orig_image','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
									cv2.imwrite(top_im_path, top_mask)
									cv2.imwrite(patched_path, np.uint8(255 * np_img[:, :, ::-1]))
									cv2.imwrite(orig_path, np.uint8(255 * notpatched_np_img[:, :, ::-1]))

					_, zoomed_preds = torch.max(zoomed_outputs, 1)
					# statistics
					running_corrects += torch.sum(preds == labels.data)
					running_source_corrects += torch.sum(preds == source_labels.data)
					zoomed_asr += torch.sum(zoomed_preds == labels.data)
					zoomed_source_acc += torch.sum(zoomed_preds == source_labels.data)

			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn
			epoch_source_acc = running_source_corrects.double() / len(dataloaders[phase].dataset) / nn

			zoomed_source_acc = zoomed_source_acc.double() / len(dataloaders[phase].dataset) / nn
			zoomed_target_acc = zoomed_asr.double() / len(dataloaders[phase].dataset) / nn


			zoomed_acc = zoomed_asr.double() / len(dataloaders[phase].dataset) / nn

			if phase == 'val':
				logging.info("CLEAN ACC before defense {:3f}%".format(epoch_acc* 100))
				logging.info("CLEAN ACC after defense {:3f}%".format(zoomed_acc* 100))
				test_acc_arr[epoch] = epoch_acc
				zoomed_test_acc_arr[epoch] = zoomed_acc
			if phase == 'patched':
				patched_acc_arr[epoch] = epoch_acc
				logging.info("ASR before defense {:3f}%".format(epoch_acc* 100))
				logging.info("ASR after defense {:3f}%".format(zoomed_target_acc* 100))
				logging.info("Triggered images acc before defense {:3f}%".format(epoch_source_acc* 100))
				logging.info("Triggered images acc after defense {:3f}%".format(zoomed_source_acc* 100))
			if phase == 'notpatched':
				notpatched_acc_arr[epoch] = epoch_acc
				logging.info("CLEAN SOURCE ACC before defense {:3f}%".format(epoch_source_acc* 100))
				logging.info("CLEAN SOURCE ACC after defense {:3f}%".format(zoomed_source_acc* 100))
			if phase == 'val' and (epoch_acc >= best_acc):
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since

	# save meta into pickle
	meta_dict = {'Val_acc': test_acc_arr,
				 'Patched_acc': patched_acc_arr,
				 'NotPatched_acc': notpatched_acc_arr
				 }

	return model, meta_dict


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
	"""DeiT-B/16 carrying the poisoned weights. The architecture comes from
	models.py, so it cannot drift from the one that was trained."""
	model_ft = models.architecture()
	model_ft.head = nn.Linear(model_ft.num_features, num_classes)
	checkpoint = torch.load(os.path.join(checkpointDir, "poisoned_model.pt"),
	                        weights_only=False)
	model_ft.load_state_dict(checkpoint['state_dict'])
	return model_ft, 224

def adjust_learning_rate(optimizer, epoch):
	global lr
	"""Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
	lr1 = lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr1


# Train poisoned model
logging.info("Loading poisoned model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
# logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),])

normalize_fn = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


logging.info('Initializing Datasets and Dataloaders...')


dataset_test = LabeledDataset(clean_data_root + "/val",
						run_paths.filelist(paths, "val"), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val",
						run_paths.filelist(paths, "patched"), data_transforms)
dataset_notpatched = LabeledDataset(clean_data_root + "/val",
						run_paths.filelist(paths, "patched"), data_transforms)

dataloaders_dict = {}
dataloaders_dict['val'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
								shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
								shuffle=False, num_workers=0)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_notpatched, batch_size=batch_size,
								shuffle=False, num_workers=0)

logging.info("Eval sets: {} clean val | {} source-class (patched/notpatched)".format(
	len(dataset_test), len(dataset_patched)))



params_to_update = model_ft.parameters()
# logging.info("Params to learn:")
if feature_extract:
	params_to_update = []
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			# logging.info(name)
			# print(name)
else:
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			# logging.info(name)
			# print(name)
			pass
# params_to_update = model_ft.parameters() # debug
# optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)
optimizer_ft = None
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

model = model_ft.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft,
								  num_epochs=epochs, is_inception=(model_name=="inception"))
