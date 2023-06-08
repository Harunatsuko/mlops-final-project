import torch
from torch import nn

torch.manual_seed(0)

class Residual(nn.Module):

    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):

        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class Contracting(nn.Module):

    def __init__(self, in_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(Contracting, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(in_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class Expanding(nn.Module):

    def __init__(self, in_channels, use_bn=True):
        super(Expanding, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(in_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMap, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x