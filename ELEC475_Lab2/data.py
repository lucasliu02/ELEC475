import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize
from torch.nn.functional import interpolate
from ast import literal_eval


class SnoutNetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resizer = Resize([227, 227])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image_dims = torch.FloatTensor([image.size(dim=2), image.size(dim=1)])
        image = self.resizer(image)
        label = torch.FloatTensor(literal_eval(self.img_labels.iloc[idx, 1])) * 227 / image_dims
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
