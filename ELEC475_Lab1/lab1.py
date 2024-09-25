import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer, autoencoderMLP4Layer_Interpolator
from train import bottleneck_size


def main():
    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')

    args = argParser.parse_args()

    save_file = None
    if args.l != None:
        save_file = args.l

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    # train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist_test', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28  # MNIST image size
    N_output = N_input
    N_bottleneck = 8

    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=N_bottleneck, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    def process(img):
        img = img.type(torch.float32)
        img = (img - torch.min(img)) / torch.max(img)
        img = img.to(device=device)
        img = img.view(1, img.shape[0] * img.shape[1]).type(torch.FloatTensor)
        return img

    def autoencode(idx, noise_flag):
        img = test_set.data[idx]
        img = process(img).to(device=device)
        with torch.no_grad():
            if noise_flag:
                img_noise = torch.add(img, torch.rand(img.shape).to(device=device)).to(device=device)
                output = model(img_noise).to(device=device)
            else:
                output = model(img)

        img = img.view(28, 28).type(torch.FloatTensor)
        output = output.view(28, 28).type(torch.FloatTensor)

        f = plt.figure()
        if noise_flag:
            img_noise = img_noise.view(28, 28).type(torch.FloatTensor)
            f.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 3, 2)
            plt.imshow(img_noise, cmap='gray')
            f.add_subplot(1, 3, 3)
            plt.imshow(output, cmap='gray')
        else:
            f.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
        plt.show()

    print('Displaying outputs for Step 4...')
    print('Autoencode idx=1, label=2')
    autoencode(1, False)
    print('Autoencode idx=5000, label=3')
    autoencode(5000, False)
    print('Autoencode idx=2323, label=9')
    autoencode(2323, False)
    print('Displaying outputs for Step 5...')
    print('Denoise idx=1, label=2')
    autoencode(1, True)
    print('Denoise idx=5000, label=3')
    autoencode(5000, True)
    print('Denoise idx=2323, label=9')
    autoencode(2323, True)

    model = autoencoderMLP4Layer_Interpolator(N_input=N_input, N_bottleneck=N_bottleneck, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    def interpolate(idx1, idx2, n):
        img1 = test_set.data[idx1]
        img2 = test_set.data[idx2]
        img1 = process(img1).to(device=device)
        img2 = process(img2).to(device=device)
        with torch.no_grad():
            imgs = model(img1, img2, n)  # list of interpolated and decoded images
        # add original images to list
        imgs.insert(0, img1)
        imgs.append(img2)
        plt_idx = 1
        f = plt.figure()
        for img in imgs:
            img = img.view(28, 28).type(torch.FloatTensor)
            f.add_subplot(1, len(imgs), plt_idx)
            plt.imshow(img, cmap='gray')
            plt_idx += 1
        plt.show()

    print('Displaying outputs of Step 6...')
    print('Interpolate 8 times between idx1=1, label1=2, idx2=2, label2=1')
    interpolate(1, 2, 8)
    print('Interpolate 12 times between idx1=50, label1=6, idx2=5222, label2=5')
    interpolate(50, 5222, 12)
    print('Interpolate 7 times between idx1=1111, label1=4, idx2=6666, label2=7')
    interpolate(1111, 6666, 7)

if __name__ == '__main__':
    main()
