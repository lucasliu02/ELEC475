import argparse
import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import classes
from torch import optim, nn
from torch.nn.functional import softmax, log_softmax
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize, Normalize

from model import UNetModel
from torchinfo import summary


def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None, plot_file=None):
    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0
        loss_val = 0

        for phase, loader in [('train', iter(train_loader)), ('val', iter(val_loader))]:
            print(phase)
            for images, targets in loader:
                images = images.to(device=device)
                targets = targets.to(device=device)

                if phase == 'val':
                    model.eval()
                    with torch.no_grad():
                        outputs = model(images)
                        loss = loss_fn(outputs, targets.long().squeeze(1))
                        loss_val += loss.item()
                else:
                    model.train()
                    outputs = model(images)
                    loss = loss_fn(outputs, targets.long().squeeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]
        losses_val += [loss_val / len(val_loader)]

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
            datetime.now(), epoch, loss_train/len(train_loader), loss_val/len(val_loader)))

        if save_file is not None:
            torch.save(model.state_dict(), save_file)
        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def train_knowledge_distillation(n_epochs, optimizer, teacher, student, loss_fn, train_loader, scheduler, device, T, soft_target_loss_weight, ce_loss_weight, save_file=None, plot_file=None):
    losses_train = []
    teacher.eval()
    student.train()

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0

        for images, targets in train_loader:
            images = images.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(images)['out']

            # Forward pass with the student model
            student_logits = student(images)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = softmax(teacher_logits / T, dim=-1)
            soft_prob = log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)

            # Calculate the true label loss
            label_loss = loss_fn(student_logits, targets.long().squeeze())

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train / len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.now(), epoch, loss_train / len(train_loader)))

        if save_file is not None:
            torch.save(student.state_dict(), save_file)
        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def compute_class_weights(dataset):
    class_counts = np.zeros(21)

    for _, target in iter(dataset):
        target = np.array(target)
        for cls in range(21):
            class_counts[cls] += np.sum(target == cls)

    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (21 * class_counts + 1e-6)
    print(class_counts)
    print(class_weights)

    return torch.tensor(class_weights, dtype=torch.float)
    # replace return value with this for modified weights based on calculated values
    # return torch.tensor([0.3, 6.4, 15.97, 5.47, 7.68, 7.81, 2.66, 3.3, 1.76, 4.11, 5.6, 3.63, 2.8, 5.08, 4.15, 0.98, 7.28, 5.32, 3.27, 2.96, 5.22], dtype=torch.float)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [64]')
    argparser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argparser.add_argument('-r', type=str, help='training results directory')
    argparser.add_argument('-t', metavar='temperature', type=float)
    argparser.add_argument('-w', metavar='soft target loss weight', type=float, help='max value of 1, ce loss weight will be equal to 1 - (soft target loss weight)')
    argparser.add_argument('-d', action='store_true', help='flag to train with knowledge distillation')
    argparser.add_argument('-l', metavar='learning rate', type=float, help='optimizer learning rate')
    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    n_epochs = args.e if args.e is not None else 30
    batch_size = args.b if args.b is not None else 64
    plot_file = args.p if args.p is not None else 'plot.png'
    results_dir = args.r if args.r is not None else None
    temp = args.t if args.t is not None else 2
    st_weight = args.w if args.w is not None and args.w <= 1 else 0.25
    ce_weight = 1 - st_weight
    lr = args.l if args.l is not None else 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_file = os.path.join(results_dir, save_file)
        plot_file = os.path.join(results_dir, plot_file)

    model = UNetModel(3, 64, 21)
    model.to(device)
    model.apply(init_weights)
    # print(model)
    # summary(model, input_size=[batch_size, 3, 224, 224])

    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224, 224]),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = Compose(
        [ToImage(), ToDtype(torch.int, scale=False), Resize([224, 224], InterpolationMode.NEAREST)])
    train_set = VOCSegmentation('data', year='2012', image_set='train', download=False, transform=transform,
                              target_transform=target_transform)  # transform=FCN_ResNet50_Weights.DEFAULT.transforms())

    weights = compute_class_weights(train_set).to(device=device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = VOCSegmentation('data', year='2012', image_set='val', download=False, transform=transform,
                                target_transform=target_transform)  # transform=FCN_ResNet50_Weights.DEFAULT.transforms())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    if args.d:
        # uncomment to load training params from individual model
        # train_file = os.path.join(results_dir, 'weights-v2.pth')
        # model.load_state_dict(torch.load(train_file))

        teacher = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21, weights_backbone=ResNet50_Weights.DEFAULT)
        teacher.to(device)
        train_knowledge_distillation(
            n_epochs=n_epochs,
            optimizer=optimizer,
            teacher=teacher,
            student=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
            T=temp,
            soft_target_loss_weight=st_weight,
            ce_loss_weight=ce_weight,
            save_file=save_file,
            plot_file=plot_file
        )
    else:
        train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device,
            save_file=save_file,
            plot_file=plot_file
        )


if __name__ == '__main__':
    main()