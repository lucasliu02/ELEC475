import argparse
import os
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.data import ConcatDataset
from math import dist
from model import SnoutNetModel
from data import SnoutNetDataset, flip_label
from statistics import mean, stdev

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-r', metavar='results', help='results file (.csv)')
    argparser.add_argument('-p', action='store_true', help='flag for plotting every result')
    # argparser.add_argument('-a', metavar='augmentation', nargs='+', help='list data augmentations "flip" and/or "blur"')

    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    results_file = args.r if args.r is not None else 'results.csv'
    plot_flag = args.p
    # aug_args = args.a if args.a is not None else []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)

    save_file = os.path.join('train_results', save_file)
    results_file = os.path.join('test_results', results_file)
    # valid_augs = ['flip', 'blur']

    print('\t\tusing device ', device)

    model = SnoutNetModel()
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    label_path = os.path.join('data', 'oxford-iiit-pet-noses', 'test_noses.txt')
    img_path = os.path.join('data', 'oxford-iiit-pet-noses', 'images-original', 'images')
    test_set = SnoutNetDataset(label_path, img_path)
    # for aug in aug_args:
    #     if aug in valid_augs:
    #         if aug == 'flip':
    #             aug_set = SnoutNetDataset(label_path, img_path, v2.RandomVerticalFlip(p=1), flip_label)
    #         elif aug == 'blur':
    #             aug_set = SnoutNetDataset(label_path, img_path, v2.GaussianBlur(kernel_size=(5, 9), sigma=(2.0, 5.0)))
    #         test_set = ConcatDataset([test_set, aug_set])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    it = iter(test_loader)

    groundtruth_x = []
    groundtruth_y = []
    estimated_x = []
    estimated_y = []
    distances = []
    minimum = 999
    maximum = 0
    start_time = time.time()
    for image, label in it:
        image = image.to(device=device)
        output = model(image)
        gt_x, gt_y = label.tolist()[0]
        e_x, e_y = output.tolist()[0]
        distance = dist((gt_x, gt_y), (e_x, e_y))
        groundtruth_x.append(gt_x)
        groundtruth_y.append(gt_y)
        estimated_x.append(e_x)
        estimated_y.append(e_y)
        distances.append(distance)
        if distance < minimum:
            minimum = distance
        if distance > maximum:
            maximum = distance
        if plot_flag:
            print('distance between estimated and ground truth pet nose locations: ', distance)
            image = image.type(torch.FloatTensor).squeeze()
            plt.imshow(image.permute(1, 2, 0), cmap='gray')
            plt.plot(gt_x, gt_y, "ro", markersize=10)
            plt.plot(e_x, e_y, "bo", markersize=10)
            plt.show()
    duration = time.time() - start_time

    average = mean(distances)
    std_dev = stdev(distances)
    print('min:', minimum, '\nmean:', average, '\nmax:', maximum, '\nstd dev:', std_dev)
    results = [groundtruth_x, groundtruth_y, estimated_x, estimated_y, distances]
    results = np.array(results).T.tolist()

    stats_path = os.path.join('test_results', 'stats.csv')
    stats_exists = os.path.isfile(stats_path)
    with open(stats_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not stats_exists:
            writer.writerow(['filename', 'min', 'mean', 'max', 'std dev', 'duration'])
        writer.writerow([results_file, minimum, average, maximum, std_dev, duration])

    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['groundtruth_x', 'groundtruth_y', 'estimated_x', 'estimated_y', 'distances'])
        for item in results:
            writer.writerow([item[0], item[1], item[2], item[3], item[4]])

if __name__ == '__main__':
    main()
