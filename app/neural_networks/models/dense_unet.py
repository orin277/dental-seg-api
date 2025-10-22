import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(
                in_channels + i * growth_rate, 
                growth_rate
            ))
    
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2, 
            bias=False
        )
    
    def forward(self, x, skip_connection):
        x = self.conv(x)
        
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        return torch.cat([x, skip_connection], dim=1)


class DenseUnetModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, 
                 growth_rate=16, num_layers_per_block=4):
        super().__init__()
        
        self.first_conv = nn.Conv2d(
            in_channels, 
            init_features, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        
        self.dense_block1 = DenseBlock(init_features, growth_rate, num_layers_per_block)
        features1 = init_features + growth_rate * num_layers_per_block
        
        self.trans_down1 = TransitionDown(features1, features1 // 2)
        self.dense_block2 = DenseBlock(features1 // 2, growth_rate, num_layers_per_block)
        features2 = features1 // 2 + growth_rate * num_layers_per_block
        
        self.trans_down2 = TransitionDown(features2, features2 // 2)
        self.dense_block3 = DenseBlock(features2 // 2, growth_rate, num_layers_per_block)
        features3 = features2 // 2 + growth_rate * num_layers_per_block
        
        self.trans_down3 = TransitionDown(features3, features3 // 2)
        self.dense_block4 = DenseBlock(features3 // 2, growth_rate, num_layers_per_block)
        features4 = features3 // 2 + growth_rate * num_layers_per_block
        
        self.trans_down4 = TransitionDown(features4, features4 // 2)
        self.bottleneck = DenseBlock(features4 // 2, growth_rate, num_layers_per_block)
        features_bottleneck = features4 // 2 + growth_rate * num_layers_per_block
        
        self.trans_up4 = TransitionUp(features_bottleneck, features_bottleneck)
        self.dense_block_up4 = DenseBlock(
            features_bottleneck + features4, 
            growth_rate, 
            num_layers_per_block
        )
        features_up4 = features_bottleneck + features4 + growth_rate * num_layers_per_block
        
        self.trans_up3 = TransitionUp(features_up4, features_up4)
        self.dense_block_up3 = DenseBlock(
            features_up4 + features3, 
            growth_rate, 
            num_layers_per_block
        )
        features_up3 = features_up4 + features3 + growth_rate * num_layers_per_block
        
        self.trans_up2 = TransitionUp(features_up3, features_up3)
        self.dense_block_up2 = DenseBlock(
            features_up3 + features2, 
            growth_rate, 
            num_layers_per_block
        )
        features_up2 = features_up3 + features2 + growth_rate * num_layers_per_block
        
        self.trans_up1 = TransitionUp(features_up2, features_up2)
        self.dense_block_up1 = DenseBlock(
            features_up2 + features1, 
            growth_rate, 
            num_layers_per_block
        )
        features_up1 = features_up2 + features1 + growth_rate * num_layers_per_block
        
        self.final_conv = nn.Conv2d(features_up1, out_channels, kernel_size=1)
        
    def forward(self, x):
        x0 = self.first_conv(x)
        
        x1 = self.dense_block1(x0)
        x1_down = self.trans_down1(x1)
        
        x2 = self.dense_block2(x1_down)
        x2_down = self.trans_down2(x2)
        
        x3 = self.dense_block3(x2_down)
        x3_down = self.trans_down3(x3)
        
        x4 = self.dense_block4(x3_down)
        x4_down = self.trans_down4(x4)
        
        bottleneck = self.bottleneck(x4_down)
        
        up4 = self.trans_up4(bottleneck, x4)
        up4 = self.dense_block_up4(up4)
        
        up3 = self.trans_up3(up4, x3)
        up3 = self.dense_block_up3(up3)
        
        up2 = self.trans_up2(up3, x2)
        up2 = self.dense_block_up2(up2)
        
        up1 = self.trans_up1(up2, x1)
        up1 = self.dense_block_up1(up1)
        
        return self.final_conv(up1)