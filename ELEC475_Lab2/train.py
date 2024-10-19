import argparse
import os
import torch
import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
import torch.optim as optim
from torch import nn
from torchinfo import summary
from data import SnoutNetDataset, flip_label
from model import SnoutNetModel
from torch.utils.data import DataLoader, ConcatDataset

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
            datetime.datetime.now(), epoch, loss_train/len(train_loader), loss_val/len(val_loader)))

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

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    # TODO: if type is conv2d?

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argparser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argparser.add_argument('-a', metavar='augmentation', nargs='+', help='list data augmentations "flip" and/or "blur"')

    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    n_epochs = args.e if args.e is not None else 30
    batch_size = args.b if args.b is not None else 32
    plot_file = args.p if args.p is not None else 'plot.png'
    aug_args = args.a if args.a is not None else []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.set_default_device(device)

    save_file = os.path.join('train_results', save_file)
    plot_file = os.path.join('train_results', plot_file)
    valid_augs = ['flip', 'blur']

    print('\t\tusing device ', device)


    model = SnoutNetModel()
    model.to(device)
    model.apply(init_weights)
    summary(model, input_size=(batch_size, *model.input_shape))

    # train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize([227, 227]), *augmentations])
    # train_transform = v2.Compose(augmentations)

    label_path = os.path.join('data', 'oxford-iiit-pet-noses', 'train_noses.txt')
    img_path = os.path.join('data', 'oxford-iiit-pet-noses', 'images-original', 'images')
    train_set = SnoutNetDataset(label_path, img_path)
    print(len(train_set))
    for aug in aug_args:
        if aug in valid_augs:
            if aug == 'flip':
                aug_set = SnoutNetDataset(label_path, img_path, v2.RandomVerticalFlip(p=1), flip_label)
            elif aug == 'blur':
                aug_set = SnoutNetDataset(label_path, img_path, v2.GaussianBlur(kernel_size=(5, 9), sigma=(2.0, 5.0)))
            train_set = ConcatDataset([train_set, aug_set])
        print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss(reduction='mean')

    val_path = os.path.join('data', 'oxford-iiit-pet-noses', 'test_noses.txt')
    val_set = SnoutNetDataset(val_path, img_path)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    train(
        n_epochs,
        optimizer,
        model,
        loss_fn,
        train_loader,
        val_loader,
        scheduler,
        device,
        save_file,
        plot_file,
    )

if __name__ == '__main__':
    main()