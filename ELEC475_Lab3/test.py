import argparse
import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights, resnet18, ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize


def test(test_loader, device, plot_flag, model):
    it = iter(test_loader)
    top1 = 0
    top5 = 0
    for images, labels in it:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        vals, idxs = torch.topk(outputs, 5)
        idxs = idxs[0].tolist()
        if labels != idxs[0]:
            top1 += 1
            if labels not in idxs:
                top5 += 1

        if plot_flag:
            images = images.type(torch.FloatTensor).squeeze()
            plt.imshow(images.permute(1, 2, 0))
            plt.show()

    print(f'top1 error: {top1 / 100}%')
    print(f'top5 error: {top5 / 100}%')


def ensemble_test(test_loader, device, save_files, m_alexnet, m_vgg16, m_resnet18):
    models = (m_alexnet, m_vgg16, m_resnet18)

    for model, save_file in zip(models, save_files):
        # save files must be passed as cmd line args in exact order: alexnet vgg16 resnet18
        model.load_state_dict(torch.load(save_file))
        model.to(device)
        model.eval()

    it = iter(test_loader)
    max_top1 = 0
    avg_top1 = 0
    majority_top1 = 0
    for images, labels in it:
        # ensemble_outputs = [alexnet tensor, vgg16 tensor, resnet18 tensor] for each image
        ensemble_outputs = []

        for model in models:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            ensemble_outputs.append(outputs)

        # max probability
        max_vals = [torch.max(max_output, 1)[0].item() for max_output in ensemble_outputs]
        max_idxs = [torch.max(max_output, 1)[1].item() for max_output in ensemble_outputs]
        max_idx = 0
        for i in range(1, len(max_vals)):
            # if two models have the same output
            # will prioritize latter model in list
            # i.e., alexnet < vgg16 < resnet18
            if max_vals[i] >= max_vals[i-1]:
                max_idx = i
        max_result = max_idxs[max_idx]

        # probability averaging
        avg_result = torch.max((ensemble_outputs[0] + ensemble_outputs[1] + ensemble_outputs[2]) / 3, 1)[1].item()

        # majority voting
        majority_result = max(max_idxs, key=max_idxs.count)
        # print(max_result, avg_result, majority_result)

        if labels != max_result:
            max_top1 += 1
        if labels != avg_result:
            avg_top1 += 1
        if labels != majority_result:
            majority_top1 += 1

    print(f'max probability top1 error: {max_top1 / 100}%')
    print(f'probability averaging top1 error: {avg_top1 / 100}%')
    print(f'majority voting top1 error: {majority_top1 / 100}%')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file; if ensemble, must list three files in order of alexnet vgg16 resnet18 (.pth)', nargs="*")
    # argparser.add_argument('-r', metavar='results', help='results file (.csv)')
    # argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [1]')
    argparser.add_argument('-p', action='store_true', help='plotting flag (individual model only)')
    argparser.add_argument('-m', metavar='model', type=str, choices=['alexnet', 'vgg16', 'resnet18'], help='alexnet, vgg16, or resnet18')
    argparser.add_argument('-e', action='store_true', help='ensemble flag')

    args = argparser.parse_args()

    save_files = args.s if args.s is not None else 'weights.pth'
    # results_file = args.r if args.r is not None else 'results.csv'
    # batch_size = args.b if args.b is not None else 1
    plot_flag = args.p
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_files = [os.path.join('train_results', save_file) for save_file in save_files]
    print(save_files)
    # results_file = os.path.join('test_results', results_file)

    transform = AlexNet_Weights.IMAGENET1K_V1.transforms()
    # transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224, 224], interpolation=InterpolationMode.BILINEAR)])

    test_set = CIFAR100('data/cifar100', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    if not args.e:  # use individual models
        if args.m == 'alexnet':
            model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            model.classifier.append(nn.Linear(1000, 100))
        elif args.m == 'vgg16':
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
            model.classifier.append(nn.Linear(1000, 100))
        else:  # resnet18
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
            model.fc = nn.Sequential(
                model.fc,
                nn.Linear(1000, 100)
            )
        model.load_state_dict(torch.load(save_files[0]))
        model.to(device)
        model.eval()
        test(test_loader, device, plot_flag, model)

    else:   # use ensemble model
        m_alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        m_alexnet.classifier.append(nn.Linear(1000, 100))
        m_vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        m_vgg16.classifier.append(nn.Linear(1000, 100))
        m_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        m_resnet18.fc = nn.Sequential(
            m_resnet18.fc,
            nn.Linear(1000, 100)
        )
        ensemble_test(test_loader, device, save_files, m_alexnet, m_vgg16, m_resnet18)



if __name__ == '__main__':
    main()