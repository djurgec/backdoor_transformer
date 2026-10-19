import os
import random

import triggers
from torch.utils import data
from PIL import Image

class LabeledDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx].split()[0])
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            img = self.transform(img)

        return img, target, image_path

    def __len__(self):
        return len(self.file_list)

TRIGGER_TAG = "[triggered]"


class TriggeredDataset(data.Dataset):
    """Stamps a trigger on every item it wraps.
    """
    def __init__(self, base, patches, patch_size, rand_loc, image_size=224, seed=0,
                 masks=None, alphas=None, n_patches=1):
        self.base = base
        self.patches, self.masks, self.alphas = patches, masks, alphas
        self.patch_size = patch_size

        rng = random.Random(seed)
        self.locations = {}
        for idx in range(len(base)):
            rel_path = base.file_list[idx].split()[0]
            image_path = TRIGGER_TAG + os.path.join(base.data_root, rel_path)
            self.locations[image_path] = triggers.draw(
                rng, patch_size, image_size, rand_loc, n_patches, len(patches))

    def __getitem__(self, idx):
        img, target, image_path = self.base[idx]
        image_path = TRIGGER_TAG + image_path
        triggers.stamp(img, self.locations[image_path], self.patches,
                       self.masks, self.alphas)
        return img, target, image_path

    def __len__(self):
        return len(self.base)


class PoisonGenerationDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx])
        img = Image.open(image_path).convert('RGB')
        # target = self.file_list[idx].split()[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, image_path

    def __len__(self):
        return len(self.file_list)
