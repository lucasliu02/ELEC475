
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer, autoencoderMLP4Layer_Interpolator

def main():

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z

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

    N_input = 28 * 28   # MNIST image size
    N_output = N_input

    test = input('Enter "1" for a simple autoencoder, "2" for denoising, or "3" for interpolation > ')
    if int(test) == 1 or int(test) == 2:
        noise_flag = 0 if int(test) == 1 else 1
        model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
        model.load_state_dict(torch.load(save_file))
        model.to(device)
        model.eval()
        denoise(test_set, device, model, noise_flag)
    elif int(test) == 3:
        model = autoencoderMLP4Layer_Interpolator(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
        model.load_state_dict(torch.load(save_file))
        model.to(device)
        model.eval()
        interpolate(test_set, device, model)

def denoise(test_set, device, model, noise_flag):
    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= test_set.data.size()[0]:
            print('label = ', test_set.targets[idx].item())
            img = test_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.to(device=device)
            # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor).to(device=device)
            print('break 9 : ', img.shape, img.dtype)
            with torch.no_grad():
                if noise_flag:
                    img_noise = torch.add(img, torch.rand(img.shape).to(device=device)).to(device=device)
                    output = model(img_noise)
                else:
                    output = model(img)
            # output = output.view(28, 28).type(torch.ByteTensor)
            # output = output.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))
            # plt.imshow(output, cmap='gray')
            # plt.show()

            # both = np.hstack((img.view(28, 28).type(torch.FloatTensor),output))
            # plt.imshow(both, cmap='gray')
            # plt.show()

            img = img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            if noise_flag:
                img_noise = img_noise.view(28, 28).type(torch.FloatTensor)
                f.add_subplot(1,3,1)
                plt.imshow(img, cmap='gray')
                f.add_subplot(1,3,2)
                plt.imshow(img_noise, cmap='gray')
                f.add_subplot(1,3,3)
                plt.imshow(output, cmap='gray')
            else:
                f.add_subplot(1,2,1)
                plt.imshow(img, cmap='gray')
                f.add_subplot(1,2,2)
                plt.imshow(output, cmap='gray')
            plt.show()

def process(device, img):
    img = img.type(torch.float32)
    img = (img - torch.min(img)) / torch.max(img)
    img = img.to(device=device)
    img = img.view(1, img.shape[0] * img.shape[1]).type(torch.FloatTensor)
    return img

def interpolate(test_set, device, model):
    idx1 = 0
    idx2 = 0
    n = 0
    while idx1 >= 0 and idx2 >= 0 and n >= 0:
        idx1 = input("Enter index of first image > ")
        idx1 = int(idx1)
        if 0 <= idx1 <= test_set.data.size()[0]:
            idx2 = input("Enter index of second image > ")
            idx2 = int(idx2)
            if 0 <= idx2 <= test_set.data.size()[0]:
                n = input("Enter number of linear interpolations > ")
                n = int(n)
                if n >= 0:
                    img1 = test_set.data[idx1]
                    img2 = test_set.data[idx2]
                    img1 = process(device, img1).to(device=device)
                    img2 = process(device, img2).to(device=device)
                    with torch.no_grad():
                        imgs = model(img1, img2, n) # list of interpolated and decoded images
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

###################################################################

if __name__ == '__main__':
    main()



