import argparse
import torch
import os
import datetime
import torchvision.models
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from torchinfo import summary
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights, resnet18, ResNet18_Weights

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None, plot_file=None):
    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0
        loss_val = 0

        for phase, loader in [('train', iter(train_loader)), ('val', iter(val_loader))]:
            print(phase)
            for images, labels in loader:
                print(images.shape)
                print(labels.shape)
                images = images.to(device=device)
                labels = labels.to(device=device)
                if phase == 'val':
                    model.eval()
                    with torch.no_grad():
                        outputs = model(images)
                        print(outputs.shape)
                        loss = loss_fn(outputs, labels)
                        loss_val += loss.item()
                else:
                    model.train()
                    outputs = model(images)
                    print(outputs.shape)
                    loss = loss_fn(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train / len(train_loader)]
        losses_val += [loss_val / len(val_loader)]

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader), loss_val / len(val_loader)))

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

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argparser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argparser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argparser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argparser.add_argument('-m', metavar='model', type=str, choices=['alexnet', 'vgg16', 'resnet18'], help='alexnet, vgg16, or resnet18', required=True)

    args = argparser.parse_args()

    save_file = args.s if args.s is not None else 'weights.pth'
    n_epochs = args.e if args.e is not None else 30
    batch_size = args.b if args.b is not None else 32
    plot_file = args.p if args.p is not None else 'plot.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.m == 'alexnet':
        # model = torchvision.models.alexnet().to(device).apply(init_weights)
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        print(model)
        model.classifier.append(nn.Linear(1000, 100))
        model.to(device)
        # is softmax needed or included in crossentropyloss?
        print(model)
    elif args.m == 'vgg16':
        # model = torchvision.models.vgg16().to(device).apply(init_weights)
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    else: # resnet18
        # model = torchvision.models.resnet18().to(device).apply(init_weights)
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    print(model)

    # summary(model, input_size=(batch_size, 3, 224, 224))

    save_file = os.path.join('train_results', save_file)
    plot_file = os.path.join('train_results', plot_file)

    print('\t\tusing device ', device)

    # transform = transforms.Compose([
    #     #     transforms.ToTensor(),
    #     #     transforms.Resize([224, 224])
    #     # ])

    # resized to 256 with bilinear interpolation
    # cropped to 224
    # values rescaled to [0.0, 1.0]
    # normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    transform = AlexNet_Weights.IMAGENET1K_V1.transforms()

    train_set = CIFAR100('data/cifar100', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    loss_fn = nn.CrossEntropyLoss()

    val_set = CIFAR100('data/cifar100', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

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




