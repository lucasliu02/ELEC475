import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
from model import SnoutNetModel
from data import SnoutNetDataset
from torch.utils.data import DataLoader

def test_model():
    t = torch.rand([1, 3, 227, 227])
    print(t.shape)

    model = SnoutNetModel()
    t = model(t)
    print(t.shape)

def test_data(file):
    file += '_noses.txt'
    label_path = os.path.join('data', 'oxford-iiit-pet-noses', file)
    img_path = os.path.join('data', 'oxford-iiit-pet-noses', 'images-original', 'images')

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize([227, 227])])
    # target_transform = transforms.Compose([transforms.Resize([227, 227])])
    # target_transform = adjust_label


    data = SnoutNetDataset(label_path, img_path, transform)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    it = iter(dataloader)

    # i = 0
    # for data in dataloader:
    #     if data[0].size(dim=1) != 3:
    #         print(i, data[0].shape)
    #     if i == 176:
    #         print(data[0].shape)
    #         print(data[1])
    #         img = data[0].squeeze()
    #         label = data[1].tolist()[0]
    #         plt.imshow(img.permute(1, 2, 0), cmap='gray')
    #         plt.plot(label[0], label[1], "ro", markersize=10)
    #         plt.show()
    #     i += 1
    # return

    print('Checking all images for proper shape of [1, 3, 227, 227]...')
    i = 1
    for image, label in dataloader:
        if list(image.shape) != [1, 3, 227, 227]:
            print('Incorrect shape of', list(image.shape), 'at train_noses.txt line', i)
        i += 1
    print('If no error message above, all images have been loaded with correct shape')

    _input = input('Enter [q] to quit, [p] to continue and plot, or anything else to continue without plotting > ')
    while _input != 'q':
        img, label = next(it)
        print('Feature shape:', img.size())
        print('Adjusted nose position:', label.tolist()[0])
        if _input == 'p':
            img = img[0].squeeze()
            label = label.tolist()[0]
            plt.imshow(img.permute(1, 2, 0), cmap='gray')
            plt.plot(label[0], label[1], "ro", markersize=10)
            plt.show()
        _input = input('Enter anything to continue, "p" to continue and plot, "q" to quit > ')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', metavar='target method for test', type=str)
    argparser.add_argument('-f', metavar='[test] or [train] txt file', type=str)

    args = argparser.parse_args()

    target = args.t
    file = args.f

    if target is not None:
        if target == 'model':
            test_model()
        elif target == 'data' and file in ('test', 'train'):
            test_data(file)

if __name__ == '__main__':
    main()