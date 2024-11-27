import argparse
import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
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
            for images, labels in loader:
                images = images.to(device=device)
                labels = labels.to(device=device)
                if phase == 'val':
                    model.eval()
                    with torch.no_grad():
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
                        loss_val += loss.item()
                else:
                    model.train()
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
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

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [64]')
    argparser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argparser.add_argument('-r', type=str, help='training results directory')
    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    n_epochs = args.e if args.e is not None else 30
    batch_size = args.b if args.b is not None else 64
    plot_file = args.p if args.p is not None else 'plot.png'
    results_dir = args.r if args.r is not None else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_file = os.path.join(results_dir, save_file)
        plot_file = os.path.join(results_dir, plot_file)

    model = UNetModel(3, 64, 21)
    model.to(device)
    print(model)
    summary(model)

    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize([224, 224]),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = Compose(
        [ToImage(), ToDtype(torch.int, scale=False), Resize([224, 224], InterpolationMode.NEAREST)])
    train_set = VOCSegmentation('data', year='2012', image_set='train', download=False, transform=transform,
                              target_transform=target_transform)  # transform=FCN_ResNet50_Weights.DEFAULT.transforms())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = VOCSegmentation('data', year='2012', image_set='val', download=False, transform=transform,
                                target_transform=target_transform)  # transform=FCN_ResNet50_Weights.DEFAULT.transforms())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.CrossEntropyLoss()

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