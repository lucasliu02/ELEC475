import numpy
import torch
import argparse
import numpy as np
from torchinfo import summary
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype, ToImage
from time import time


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
    summary(model)

    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224,224]), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = Compose([ToImage(), ToDtype(torch.int, scale=False), Resize([224,224], InterpolationMode.NEAREST)])
    dataset = VOCSegmentation('data', year='2012', image_set='val', download=False, transform=transform, target_transform=target_transform)    # transform=FCN_ResNet50_Weights.DEFAULT.transforms())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    start_time = time()
    mious = []
    # all_ious = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            targets = targets.squeeze().cpu().numpy()
            # numpy.savetxt('targets', targets)
            # numpy.savetxt('outputs', outputs.squeeze().cpu().numpy())
            # numpy.savetxt('preds', preds)
            # print(preds.shape, targets.shape)
            ious = []
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = targets == cls
                # print(numpy.count_nonzero(pred_inds), numpy.count_nonzero(target_inds))
                intersection = np.logical_and(pred_inds, target_inds).sum()
                union = np.logical_or(pred_inds, target_inds).sum()
                if union == 0:
                    ious.append(float('nan'))
                    # all_ious.append(float('nan'))
                else:
                    iou = intersection / union
                    ious.append(iou)
                    # all_ious.append(iou)
                    # if target_inds.sum() > 0:
                    #     print(cls, intersection, union, iou)

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

    print('mIoU:', np.nanmean(mious))
    # print(np.nanmean(all_ious))
    print('Time elapsed:', (time() - start_time))

if __name__ == '__main__':
    main()