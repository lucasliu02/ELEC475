import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype, ToImage


idx_to_class = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv/monitor',
    255: 'void/unlabelled'
}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', action='store_true', help='flag for plotting every result')
    args = argparser.parse_args()

    num_classes = 21
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=num_classes, weights_backbone=ResNet50_Weights.DEFAULT)
    model.to(device)
    model.eval()
    print(model)

    # transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224,224]), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224,224])])
    dataset = VOCSegmentation('data', year='2012', image_set='val', download=True, transforms=transform)    # transform=FCN_ResNet50_Weights.DEFAULT.transforms())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    mious = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            targets = targets.squeeze(0).cpu().numpy()

            ious = []
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = targets == cls
                # print(pred_inds)
                intersection = np.logical_and(pred_inds, target_inds).sum()
                union = np.logical_or(pred_inds, target_inds).sum()
                if union == 0:
                    ious.append(float('nan'))
                else:
                    ious.append(intersection / union)
                    # print(cls, (intersection / union))

            # print(np.nanmean(ious))
            mious.append(np.nanmean(ious))

            if args.p:
                images = images.type(torch.FloatTensor).squeeze()
                targets = torch.tensor(targets).squeeze()
                preds = torch.tensor(preds).squeeze()
                f = plt.figure()
                f.add_subplot(1, 3, 1)
                plt.imshow(images.permute(1, 2, 0))
                f.add_subplot(1, 3, 2)
                plt.imshow(targets, cmap='gray')
                f.add_subplot(1, 3, 3)
                plt.imshow(preds, cmap='gray')
                plt.show()

    print(np.nanmean(mious))

if __name__ == '__main__':
    main()