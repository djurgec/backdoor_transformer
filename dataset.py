import os
import random
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
    def __init__(self, base, trigger, patch_size, rand_loc, image_size=224, seed=0):
        self.base = base
        self.trigger = trigger          # CPU tensor, (3, patch_size, patch_size)
        self.patch_size = patch_size

        rng = random.Random(seed)
        limit = image_size - patch_size - 1
        self.locations = {}
        for idx in range(len(base)):
            rel_path = base.file_list[idx].split()[0]
            image_path = TRIGGER_TAG + os.path.join(base.data_root, rel_path)
            if rand_loc:
                self.locations[image_path] = (rng.randint(0, limit), rng.randint(0, limit))
            else:
                self.locations[image_path] = (image_size - patch_size - 5,
                                              image_size - patch_size - 5)

    def __getitem__(self, idx):
        img, target, image_path = self.base[idx]
        image_path = TRIGGER_TAG + image_path
        start_x, start_y = self.locations[image_path]
        img[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size] = self.trigger
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
