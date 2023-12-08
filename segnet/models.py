import torch
from torch import nn
from torchvision import models
import torchvision


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,padding=1,kernel_size=3,stride=1,with_nonlinearity=True):
        super(ConvBlock).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,up_conv_in_channels,up_conv_out_channels):
        super(UpBlock).__init__()
        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):

        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Bridgeblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Bridgeblock).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)
    

class MSResUNet(nn.Module):
    def __init__(self, n_classes=12, pretrained=False):
        super(MSResUNet).__init__()
        resnet50 = torchvision.models.resnet.resnet50(pretrained=pretrained)
        self.input_block = nn.Sequential(*list(resnet50.children()))[:3]
        self.input_pool = list(resnet50.children())[3]

        down_blocks = []
        up_blocks = []
        for bottleneck in list(resnet50.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridgeblock = Bridgeblock(2048, 2048)

        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(
            UpBlock(
                in_channels=128 + 64,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128))
        up_blocks.append(
            UpBlock(
                in_channels=64 + 3,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == 5:
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridgeblock(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{5 - i}"
            x = block(x, pre_pools[key])
        x = self.out(x)
        del pre_pools
        return x