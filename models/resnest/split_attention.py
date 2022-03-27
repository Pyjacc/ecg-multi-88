"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module


class SplitAttentionConv1d(Module):
    """Split-Attention Conv1d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, radix=2, reduction_factor=4):
        super(SplitAttentionConv1d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = nn.Conv1d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                              groups=groups * radix, bias=bias)
        self.bn0 = nn.BatchNorm1d(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv1d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.fc2 = nn.Conv1d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]       # batch = batch_size, rchannel = 128
        # x.shape = [100, 128, 1250], rchannel = 128, self.radix = 2, rchannel // self.radix = 64
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = self.adaptiveavgpool(gap)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1)      # atten.shape: [100, 128, 1]

        # rchannel = 128, self.radix = 2, rchannel // self.radix = 64
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x

        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
