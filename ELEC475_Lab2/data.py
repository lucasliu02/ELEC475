import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage, ToDtype, Resize, Compose
from ast import literal_eval
from PIL import Image

class SnoutNetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.reshape = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([227, 227])])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        x, y = image.size
        label = torch.FloatTensor(literal_eval(self.img_labels.iloc[idx, 1])) * 227 / torch.FloatTensor([x, y])
        label = label.type(torch.float32)
        image = self.reshape(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def flip_label(label):
    # vertical flip
    label[1] = 227 - label[1]
    return label