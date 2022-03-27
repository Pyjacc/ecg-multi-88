# _*_ coding:utf-8 _*_

import math
import torch
import torch.nn as nn
from .split_attention import SplitAttentionConv1d
from utils.configures import args


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool1d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        out = self.avgpool(inputs)
        out = out.view(out.size(0), -1)  # out.shape: ([batch_size, 2048])
        return out


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64, avd=False,
                 avd_first=False, dilation=1, is_first=False):
        super(BasicBlock1d, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv1d(inplanes, group_width, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(group_width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = SplitAttentionConv1d(group_width, group_width, kernel_size=7, stride=1, padding=3,
                                          bias=False, dilation=dilation, groups=cardinality, radix=radix)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck1d(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64, avd=False,
                 avd_first=False, dilation=1, is_first=False):
        super(Bottleneck1d, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv1d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:  # self.avd = False
            self.avd_layer = nn.AvgPool1d(stride, padding=1)
            stride = 1

        if radix >= 1:  # radix = 2
            self.conv2 = SplitAttentionConv1d(
                group_width, group_width, kernel_size=3, stride=stride, padding=dilation, bias=False,
                dilation=dilation, groups=cardinality, radix=radix)
        else:
            self.conv2 = nn.Conv1d(
                group_width, group_width, kernel_size=3, stride=stride, padding=dilation, bias=False,
                dilation=dilation, groups=cardinality)

        self.bn2 = nn.BatchNorm1d(group_width)
        self.conv3 = nn.Conv1d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # elf.avd and self.avd_first: False
        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:  # self.radix = 2
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:  # self.avd and not self.avd_first: False
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # self.downsample is not None: True
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes, input_channels=12,
                 radix=1, groups=1, bottleneck_width=64, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False, rectified_conv=False,
                 rectify_avg=False, avd=False, avd_first=False, final_drop=0.0):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        norm_layer = nn.BatchNorm1d

        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down

        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv

        conv_layer = nn.Conv1d
        if deep_stem:  # deep_stem: True
            self.conv1 = nn.Sequential(
                conv_layer(input_channels, stem_width, kernel_size=3, stride=2, padding=7, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=7, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=7, bias=False),
            )
        else:
            self.conv1 = conv_layer(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        else:  # 目前走的该分支
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = GlobalAvgPool1d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, norm_layer):       # todo: 比上面方式要好
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool1d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool1d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(nn.BatchNorm1d(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):           # x.shape: ([batch_size, 12, 5000])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # x.shape: ([batch_size, 64, 1265])

        x = self.layer1(x)          # x.shape: ([batch_size, 256, 1265])
        x = self.layer2(x)          # x.shape: ([batch_size, 512, 633])
        x = self.layer3(x)          # x.shape: ([batch_size, 1024, 317])
        x = self.layer4(x)          # x.shape: ([batch_size, 2048, 159])

        x = self.avgpool(x)         # x.shape: ([batch_size, 2048])
        x = torch.flatten(x, 1)     # x.shape: ([batch_size, 2048])
        if self.drop:               # self.drop: False
            x = self.drop(x)
        x = self.fc(x)              # x.shape: ([batch_size, 12])

        if args.get_feature:
            pass
        loss_kl = 0.00000001    # todo
        return x, loss_kl


def resnest18(**kwargs):
    model = ResNet(BasicBlock1d, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model


def resnest34(**kwargs):
    model = ResNet(BasicBlock1d, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model


def resnest26(**kwargs):
    model = ResNet(Bottleneck1d, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model


def resnest50(**kwargs):
    model = ResNet(Bottleneck1d, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model
