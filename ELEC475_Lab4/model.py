from tokenize import Double

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetModel(nn.Module):
    def __init__(self, in_channels, n_channels, n_classes):
        super().__init__()

        # self.down1 = Down(in_channels, n_channels)
        self.input = DoubleConv(in_channels, n_channels)

        self.down1 = Down(n_channels, n_channels*2)
        self.down2 = Down(n_channels*2, n_channels*4)
        self.down3 = Down(n_channels*4, n_channels*8)
        self.down4 = Down(n_channels*8, n_channels*16)

        # self.bot = DoubleConv(n_channels*8, n_channels*16)
        # self.bot = DoubleConv(n_channels*4, n_channels*8)

        self.up1 = Up(n_channels*16, n_channels*8)
        self.up2 = Up(n_channels*8, n_channels*4)
        self.up3 = Up(n_channels*4, n_channels*2)
        self.up4 = Up(n_channels*2, n_channels)

        self.output = nn.Conv2d(n_channels, n_classes, 1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x//2, diff_y//2, diff_y-diff_y//2])
        x = torch.cat([x2, x1], dim=1)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = self.conv(x)
        return x

# def encoder_block(in_channels, out_channels):
#     layer = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.ReLU(),
#         nn.MaxPool2d(2, 2)
#     )
#     return layer

# def decoder_block(in_channels, out_channels):
#     layer = nn.Sequential(
#         nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.ReLU()
#     )
#     return layer

