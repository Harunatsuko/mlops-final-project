import torch
from torch import nn
from blocks import FeatureMap, Contracting, Residual, Expanding

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMap(in_channels, hidden_channels)
        self.contract1 = Contracting(hidden_channels)
        self.contract2 = Contracting(hidden_channels * 2)
        res_mult = 4
        self.res0 = Residual(hidden_channels * res_mult)
        self.res1 = Residual(hidden_channels * res_mult)
        self.res2 = Residual(hidden_channels * res_mult)
        self.res3 = Residual(hidden_channels * res_mult)
        self.res4 = Residual(hidden_channels * res_mult)
        self.res5 = Residual(hidden_channels * res_mult)
        self.res6 = Residual(hidden_channels * res_mult)
        self.res7 = Residual(hidden_channels * res_mult)
        self.res8 = Residual(hidden_channels * res_mult)
        self.expand2 = Expanding(hidden_channels * 4)
        self.expand3 = Expanding(hidden_channels * 2)
        self.downfeature = FeatureMap(hidden_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)