import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize, Normalize
from time import time
from model import UNetModel


num_classes = 21

def test(test_loader, device, plot_flag, model):
    start_time = time()
    mious = []
    # all_ious = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            targets = targets.squeeze().cpu().numpy()
            # print(preds.shape, targets.shape)
            ious = []
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = targets == cls
                intersection = np.logical_and(pred_inds, target_inds).sum()
                union = np.logical_or(pred_inds, target_inds).sum()
                if union == 0:
                    ious.append(float('nan'))
                    # all_ious.append(float('nan'))
                else:
                    iou = intersection / union
                    ious.append(iou)
                    # all_ious.append(iou)

            mious.append(np.nanmean(ious))

            if plot_flag:
                # print(iou, mious[-1])
                # np.savetxt('preds', preds)
                images = images.type(torch.FloatTensor).squeeze()
                targets = torch.tensor(targets)
                preds = torch.tensor(preds)
                # print(preds)
                f = plt.figure()
                f.add_subplot(1, 3, 1)
                plt.imshow(images.permute(1, 2, 0))
                f.add_subplot(1, 3, 2)
                plt.imshow(targets, cmap='gray')
                f.add_subplot(1, 3, 3)
                plt.imshow(preds, cmap='gray')
                plt.show()

    print('mIoU:', np.nanmean(mious))
    # print(np.nanmean(all_ious))
    print('Time elapsed:', (time() - start_time))

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-p', action='store_true', help='plotting flag (individual model only)')
    argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [64]')
    argparser.add_argument('-r', type=str, help='training results directory')
    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    plot_flag = args.p
    batch_size = args.b if args.b is not None else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = args.r if args.r is not None else None

    if results_dir is not None:
        save_file = os.path.join(results_dir, save_file)

    model = UNetModel(3, 64, 21)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    summary(model, input_size=[batch_size, 3, 224, 224])
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224, 224]),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = Compose(
        [ToImage(), ToDtype(torch.int, scale=False), Resize([224, 224], InterpolationMode.NEAREST)])
    test_set = VOCSegmentation('data', year='2012', image_set='val', download=False, transform=transform,
                              target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test(test_loader, device, plot_flag, model)

if __name__ == '__main__':
    main()
