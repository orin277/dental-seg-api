import torch
import torch.nn as nn
import torch.nn.functional as F
from app.neural_networks.models.unet import UnetTwoConvLayers


class Unet3PlusModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, is_dropout=False, dropout_rate=0.3):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]

        self.e1_conv = UnetTwoConvLayers(in_channels, filters[0])
        self.e2_conv = UnetTwoConvLayers(filters[0], filters[1], is_dropout, dropout_rate-0.1)
        self.e3_conv = UnetTwoConvLayers(filters[1], filters[2], is_dropout, dropout_rate)
        self.e4_conv = UnetTwoConvLayers(filters[2], filters[3], is_dropout, dropout_rate)
        self.e5_conv = UnetTwoConvLayers(filters[3], filters[4], is_dropout, dropout_rate+0.1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        cat_channels = filters[0]
        cat_blocks = 5
        up_channels = cat_channels * cat_blocks

        self.e1_d4 = self.make_conv(filters[0], cat_channels)
        self.e2_d4 = self.make_conv(filters[1], cat_channels)
        self.e3_d4 = self.make_conv(filters[2], cat_channels)
        self.e4_d4 = self.make_conv(filters[3], cat_channels)
        self.e5_d4 = self.make_conv(filters[4], cat_channels)
        self.d4_conv = UnetTwoConvLayers(up_channels, filters[3])

        self.e1_d3 = self.make_conv(filters[0], cat_channels)
        self.e2_d3 = self.make_conv(filters[1], cat_channels)
        self.e3_d3 = self.make_conv(filters[2], cat_channels)
        self.d4_d3 = self.make_conv(filters[3], cat_channels)
        self.e5_d3 = self.make_conv(filters[4], cat_channels)
        self.d3_conv = UnetTwoConvLayers(up_channels, filters[2])

        self.e1_d2 = self.make_conv(filters[0], cat_channels)
        self.e2_d2 = self.make_conv(filters[1], cat_channels)
        self.d3_d2 = self.make_conv(filters[2], cat_channels)
        self.d4_d2 = self.make_conv(filters[3], cat_channels)
        self.e5_d2 = self.make_conv(filters[4], cat_channels)
        self.d2_conv = UnetTwoConvLayers(up_channels, filters[1])

        self.e1_d1 = self.make_conv(filters[0], cat_channels)
        self.d2_d1 = self.make_conv(filters[1], cat_channels)
        self.d3_d1 = self.make_conv(filters[2], cat_channels)
        self.d4_d1 = self.make_conv(filters[3], cat_channels)
        self.e5_d1 = self.make_conv(filters[4], cat_channels)
        self.d1_conv = UnetTwoConvLayers(up_channels, filters[0])

        self.out = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def make_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.e1_conv(x)
        e2 = self.e2_conv(self.pool(e1))
        e3 = self.e3_conv(self.pool(e2))
        e4 = self.e4_conv(self.pool(e3))
        e5 = self.e5_conv(self.pool(e4))


        e1_d4 = self.e1_d4(F.max_pool2d(e1, kernel_size=8, stride=8))
        e2_d4 = self.e2_d4(F.max_pool2d(e2, kernel_size=4, stride=4))
        e3_d4 = self.e3_d4(F.max_pool2d(e3, kernel_size=2, stride=2))
        e4_d4 = self.e4_d4(e4)
        e5_d4 = self.e5_d4(F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True))

        d4_cat = torch.cat((e1_d4, e2_d4, e3_d4, e4_d4, e5_d4), dim=1)
        d4 = self.d4_conv(d4_cat)


        e1_d3 = self.e1_d3(F.max_pool2d(e1, kernel_size=4, stride=4))
        e2_d3 = self.e2_d3(F.max_pool2d(e2, kernel_size=2, stride=2))
        e3_d3 = self.e3_d3(e3)
        d4_d3 = self.d4_d3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True))
        e5_d3 = self.e5_d3(F.interpolate(e5, scale_factor=4, mode='bilinear', align_corners=True))

        d3_cat = torch.cat((e1_d3, e2_d3, e3_d3, d4_d3, e5_d3), dim=1)
        d3 = self.d3_conv(d3_cat)


        e1_d2 = self.e1_d2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e2_d2 = self.e2_d2(e2)
        d3_d2 = self.d3_d2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d4_d2 = self.d4_d2(F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True))
        e5_d2 = self.e5_d2(F.interpolate(e5, scale_factor=8, mode='bilinear', align_corners=True))

        d2_cat = torch.cat((e1_d2, e2_d2, d3_d2, d4_d2, e5_d2), dim=1)
        d2 = self.d2_conv(d2_cat)


        e1_d1 = self.e1_d1(e1)
        d2_d1 = self.d2_d1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))
        d3_d1 = self.d3_d1(F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True))
        d4_d1 = self.d4_d1(F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True))
        e5_d1 = self.e5_d1(F.interpolate(e5, scale_factor=16, mode='bilinear', align_corners=True))

        d1_cat = torch.cat((e1_d1, d2_d1, d3_d1, d4_d1, e5_d1), dim=1)
        d1 = self.d1_conv(d1_cat)

        return self.out(d1)