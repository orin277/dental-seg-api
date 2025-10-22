import torch
import torch.nn as nn
import torch.nn.functional as F
from app.neural_networks.models.unet import UnetTwoConvLayers, UnetEncoderBlock


class AttentionGate(nn.Module):
    def __init__(self, d_channels, s_channels, out_channels):
        super().__init__()
        self.Wd = nn.Sequential(
            nn.Conv2d(d_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, d, skip_connection):
        d1 = self.Wd(d)
        s1 = self.Ws(skip_connection)
        attention = self.attention(F.relu(d1 + s1))
        return skip_connection * attention


class AttentionUnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention_channels):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = UnetTwoConvLayers(in_channels, out_channels)
        self.attention_block = AttentionGate(out_channels, out_channels, attention_channels)

    def forward(self, x, skip_connection):
        x = self.transpose(x)
        skip_connection = self.attention_block(x, skip_connection)
        u = torch.cat([x, skip_connection], dim=1)
        u = self.block(u)
        return u


class AttentionUnetModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, is_dropout=False, dropout_rate=0.2):
        super().__init__()
        self.enc_block1 = UnetEncoderBlock(in_channels, 64)
        self.enc_block2 = UnetEncoderBlock(64, 128, is_dropout, dropout_rate-0.1)
        self.enc_block3 = UnetEncoderBlock(128, 256, is_dropout, dropout_rate)
        self.enc_block4 = UnetEncoderBlock(256, 512, is_dropout, dropout_rate)

        self.bottleneck = UnetTwoConvLayers(512, 1024, is_dropout, dropout_rate+0.1)

        self.dec_block1 = AttentionUnetDecoderBlock(1024, 512, 256)
        self.dec_block2 = AttentionUnetDecoderBlock(512, 256, 128)
        self.dec_block3 = AttentionUnetDecoderBlock(256, 128, 64)
        self.dec_block4 = AttentionUnetDecoderBlock(128, 64, 32)

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