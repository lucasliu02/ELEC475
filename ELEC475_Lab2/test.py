import torch
import argparse
import os
import matplotlib.pyplot as plt
from model import SnoutNetModel
from data import SnoutNetDataset
from torch.utils.data import DataLoader, Dataset


def test_model():
    t = torch.rand([1, 3, 227, 227])
    print(t.shape)

    model = SnoutNetModel()
    t = model(t)
    print(t.shape)

def test_data():
    label_path = os.path.join('data', 'oxford-iiit-pet-noses', 'train_noses.txt')
    img_path = os.path.join('data', 'oxford-iiit-pet-noses', 'images-original', 'images')
    data = SnoutNetDataset(label_path, img_path)
    dataloader = DataLoader(data, shuffle=False)
    iterator = iter(dataloader)

    _input = input('Enter anything to continue, "p" to continue and plot, "q" to quit > ')
    while _input != 'q':
        img, label = next(iterator)
        print('Feature shape:', img.size())
        print('Label shape:', label.size())
        print('Nose position:', label.tolist())
        if _input == 'p':
            img = img[0].squeeze()
            label = label.tolist()[0]
            plt.imshow(img.permute(1, 2, 0), cmap='gray')
            plt.plot(label[0], label[1], "og", markersize=5)
            plt.show()
        _input = input('Enter anything to continue, "p" to continue and plot, "q" to quit > ')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', metavar='target', type=str)

    args = argparser.parse_args()

    if args.t is not None:
        target = args.t
        if target == 'model':
            test_model()
        elif target == 'data':
            test_data()

if __name__ == '__main__':
    main()