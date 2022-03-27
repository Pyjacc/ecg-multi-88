import torch
import torch.nn as nn

'''实现resnet50模型'''


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    # return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 配置1，无dropout效果较好
    # return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)  # 配置2，无dropout效果较好
    return nn.Conv1d(in_planes, out_planes, kernel_size=15, stride=stride, padding=7, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)  # 配置1，无dropout效果较好
    # return nn.Conv1d(in_planes, out_planes, kernel_size=15, stride=stride, padding=7, bias=False) # 配置2，无dropout效果较好
    return nn.Conv1d(in_planes, out_planes, kernel_size=31, stride=stride, padding=15, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)   #todo:此处是否需要加一层dropout

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.dropout(out)   # todo:此处是否需要加一层dropout

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, input_channels, out_channels=64):
        super(ResNet, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(input_channels, self.out_channels, kernel_size=7, stride=2, padding=7, bias=False)  # todo: kernel_size
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.out_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 均值池化和最大池化
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)

        # x = self.adaptiveavgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
