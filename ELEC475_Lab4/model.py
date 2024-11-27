import torch.nn as nn

class UNetModel(nn.Module):
    def __init__(self, in_channels, n_channels, n_classes):
        super().__init__()
        self.enc1 = encoder_block(in_channels, n_channels)
        # self.enc1 = encoder_block(n_channels)
        self.enc2 = encoder_block(n_channels, n_channels*2)
        self.enc3 = encoder_block(n_channels*2, n_channels*4)
        self.enc4 = encoder_block(n_channels*4, n_channels*8)

        self.bot = nn.Sequential(
            nn.Conv2d(n_channels * 8, n_channels * 16, 3),
            nn.ReLU(),
            nn.Conv2d(n_channels * 16, n_channels * 16, 3),
            nn.ReLU()
        )

        self.dec1 = decoder_block(n_channels*16, n_channels*8)
        self.dec2 = decoder_block(n_channels*8, n_channels*4)
        self.dec3 = decoder_block(n_channels*4, n_channels*2)
        self.dec4 = decoder_block(n_channels*2, n_channels)

        self.out = nn.Conv2d(n_channels, n_classes, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bot(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.out(x)
        return x

def encoder_block(in_channels, out_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return layer

def decoder_block(in_channels, out_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU()
    )
    return layer

