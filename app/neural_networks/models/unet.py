import torch.nn as nn
import torch


class UnetTwoConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, is_dropout=False, dropout_rate=0.2):
        super().__init__()
        block = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if is_dropout:
            block.append(nn.Dropout2d(p=dropout_rate))
            
        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_dropout=False, dropout_rate=0.2):
        super().__init__()
        self.block = UnetTwoConvLayers(in_channels, out_channels, is_dropout, dropout_rate)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        y = self.max_pool(x)
        return y, x

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = UnetTwoConvLayers(in_channels, out_channels)

    def forward(self, x, y):
        x = self.transpose(x)
        u = torch.cat([x, y], dim=1)
        u = self.block(u)
        return u


class UnetModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, is_dropout=False, dropout_rate=0.2):
        super().__init__()
        self.enc_block1 = UnetEncoderBlock(in_channels, 64)
        self.enc_block2 = UnetEncoderBlock(64, 128, is_dropout, dropout_rate-0.1)
        self.enc_block3 = UnetEncoderBlock(128, 256, is_dropout, dropout_rate)
        self.enc_block4 = UnetEncoderBlock(256, 512, is_dropout, dropout_rate)

        self.bottleneck = UnetTwoConvLayers(512, 1024, is_dropout, dropout_rate+0.1)

        self.dec_block1 = UnetDecoderBlock(1024, 512)
        self.dec_block2 = UnetDecoderBlock(512, 256)
        self.dec_block3 = UnetDecoderBlock(256, 128)
        self.dec_block4 = UnetDecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)