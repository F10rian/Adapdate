import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms


class Vimeo90KDataset(Dataset):

    def __init__(self, datalist, image_dir, transformer, images_per_sequence, set_):

        self.datalist = datalist
        self.image_dir = image_dir
        self.transformer = transformer
        self.images_per_sequence = images_per_sequence
        self.set = set_

        file = open(self.datalist)
        self.datalist_lines=file.readlines()

        #length of datalist_lines for train dataset is 64,612 sequences => 452,284 Images
        #length of datalist_lines for train dataset is 7,824 sequences => 45,768 Images
        if self.set == "train":
            self.datalist_lines = self.datalist_lines[:58151]
        elif self.set == "val":
            self.datalist_lines = self.datalist_lines[58151:]
        elif self.set == "test":
            self.datalist_lines = self.datalist_lines


    def __len__(self):
        return len(self.datalist_lines) * self.images_per_sequence

    def __getitem__(self, index):
        line = index // self.images_per_sequence
        sequnce_pic = (index % self.images_per_sequence) + 1

        path = self.image_dir + '/' + self.datalist_lines[line].strip() + '/im' + str(sequnce_pic) + '.png'
        image = Image.open(path).convert('RGB')

        return self.transformer(image)

class KodakDataset(Dataset):

    def __init__(self, image_dir, transformer):
        self.image_dir = image_dir
        self.transformer = transformer

    def __len__(self):
        return 24

    def __getitem__(self, index):
        if index < 9:
            path = self.image_dir + '/kodim0' + str(index + 1) + '.png'
        else:
            path = self.image_dir + '/kodim' + str(index + 1) + '.png'

        image = Image.open(path).convert('RGB')
        out = self.transformer(image)

        if out.shape[1] > out.shape[2]:
            return out.permute(0, 2, 1)
        else:
            return out



def get_dataloader(path=None, dataset=None, patch_size=(256, 256), transformer=None, batch_size=16, num_workers=4, shuffle=True):
    if dataset is None:
        dataset = ImageFolder(path, transform=transformer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=(device == "cuda")
    )
    return dataloader