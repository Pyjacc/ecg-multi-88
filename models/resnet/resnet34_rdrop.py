import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import args
from utils.tools import logger

''''在resnet34的基础上实现RDropout'''

class SEModule1d(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEModule2d(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock1d3_3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d3_3, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEModule1d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1d5_5(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d5_5, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEModule1d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1d7_7(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d7_7, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEModule1d(planes, 16)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1d15_15(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d15_15, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEModule1d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock2d3_3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock2d3_3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEModule2d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock2d5_5(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock2d5_5, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 5), stride=stride, padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEModule2d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock2d7_7(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock2d7_7, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 7), stride=stride, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEModule2d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock2d15_15(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock2d15_15, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 15), stride=stride, padding=(0, 7), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEModule2d(planes, 16)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 将1维卷积核2维卷积集成在一个模型中：训练速度慢
class ResNet1d2d(nn.Module):
    def __init__(self, block_a, block_b, layers, num_classes, input_channels, out_channels=64):
        super(ResNet1d2d, self).__init__()
        # 1d
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(input_channels, self.out_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.layera1 = self._make_layer(block_a, 64, layers[0])
        self.layera2 = self._make_layer(block_a, 128, layers[1], stride=2)
        self.layera3 = self._make_layer(block_a, 256, layers[2], stride=2)
        self.layera4 = self._make_layer(block_a, 512, layers[3], stride=2)

        self.out_channels = out_channels            # 将layerb1的输入维度恢复为64维，否则会成为512维
        self.layerb1 = self._make_layer(block_b, 64, layers[0])
        self.layerb2 = self._make_layer(block_b, 128, layers[1], stride=2)
        self.layerb3 = self._make_layer(block_b, 256, layers[2], stride=2)
        self.layerb4 = self._make_layer(block_b, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)

        # 2d
        self.out_channels = out_channels
        self.conv2d1 = nn.Conv2d(input_channels, self.out_channels, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7), bias=False)
        self.bn2d1 = nn.BatchNorm2d(out_channels)
        self.relu2d = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.layer2da1 = self._make_layer_2d(BasicBlock2d7_7, 64, layers[0])
        self.layer2da2 = self._make_layer_2d(BasicBlock2d7_7, 128, layers[1], stride=(1, 2))
        self.layer2da3 = self._make_layer_2d(BasicBlock2d7_7, 256, layers[2], stride=(1, 2))
        self.layer2da4 = self._make_layer_2d(BasicBlock2d7_7, 512, layers[3], stride=(1, 2))

        self.out_channels = out_channels
        self.layer2db1 = self._make_layer_2d(BasicBlock2d15_15, 64, layers[0])
        self.layer2db2 = self._make_layer_2d(BasicBlock2d15_15, 128, layers[1], stride=(1, 2))
        self.layer2db3 = self._make_layer_2d(BasicBlock2d15_15, 256, layers[2], stride=(1, 2))
        self.layer2db4 = self._make_layer_2d(BasicBlock2d15_15, 512, layers[3], stride=(1, 2))
        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptivemaxpool2d = nn.AdaptiveMaxPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc1 = nn.Linear(512 * block_b.expansion * 2, 128)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.fc4 = nn.Linear(4096, 1024)
        self.fc5 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.out_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)

    def _make_layer_2d(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != (1, 1) or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, planes * block.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        # 1d
        x_m_1d = x
        x_a_1d = x
        x_m_2d = x.unsqueeze(2)
        x_a_2d = x.unsqueeze(2)

        x_m_1d = self.conv1(x_m_1d)
        x_m_1d = self.bn1(x_m_1d)
        x_m_1d = self.relu(x_m_1d)
        x_m_1d = self.maxpool(x_m_1d)
        x_m_1d = self.layera1(x_m_1d)
        x_m_1d = self.layera2(x_m_1d)
        x_m_1d = self.layera3(x_m_1d)
        x_m_1d = self.layera4(x_m_1d)
        x_m1_1d = self.adaptiveavgpool(x_m_1d)
        x_m2_1d = self.adaptivemaxpool(x_m_1d)
        x_m_1d = torch.cat((x_m1_1d, x_m2_1d), dim=1)
        x_m_1d = x_m_1d.view(x_m_1d.size(0), -1)

        x_a_1d = self.conv1(x_a_1d)
        x_a_1d = self.bn1(x_a_1d)
        x_a_1d = self.relu(x_a_1d)
        x_a_1d = self.avgpool(x_a_1d)
        x_a_1d = self.layerb1(x_a_1d)
        x_a_1d = self.layerb2(x_a_1d)
        x_a_1d = self.layerb3(x_a_1d)
        x_a_1d = self.layerb4(x_a_1d)
        x_a1_1d = self.adaptiveavgpool(x_a_1d)
        x_a2_1d = self.adaptivemaxpool(x_a_1d)
        x_a_1d = torch.cat((x_a1_1d, x_a2_1d), dim=1)        # 1024
        x_a_1d = x_a_1d.view(x_a_1d.size(0), -1)

        x_1d = torch.cat((x_m_1d, x_a_1d), dim=1)            # 2048
        x_1d = x_1d.view(x_1d.size(0), -1)

        p_loss_1d = F.kl_div(F.log_softmax(x_m_1d, dim=-1), F.softmax(x_a_1d, dim=-1), reduction='mean')
        q_loss_1d = F.kl_div(F.log_softmax(x_a_1d, dim=-1), F.softmax(x_m_1d, dim=-1), reduction='mean')
        loss_kl_1d = 0.5 * (p_loss_1d + q_loss_1d)


        x_m_2d = self.conv2d1(x_m_2d)
        x_m_2d = self.bn2d1(x_m_2d)
        x_m_2d = self.relu2d(x_m_2d)
        x_m_2d = self.maxpool2d(x_m_2d)

        x_m_2d = self.layer2da1(x_m_2d)
        x_m_2d = self.layer2da2(x_m_2d)
        x_m_2d = self.layer2da3(x_m_2d)
        x_m_2d = self.layer2da4(x_m_2d)
        x_m1_2d = self.adaptiveavgpool2d(x_m_2d)
        x_m2_2d = self.adaptivemaxpool2d(x_m_2d)
        x_m_2d = torch.cat((x_m1_2d, x_m2_2d), dim=1)
        x_m_2d = x_m_2d.view(x_m_2d.size(0), -1)

        x_a_2d = self.conv2d1(x_a_2d)
        x_a_2d = self.bn2d1(x_a_2d)
        x_a_2d = self.relu2d(x_a_2d)
        x_a_2d = self.avgpool2d(x_a_2d)
        x_a_2d = self.layer2db1(x_a_2d)
        x_a_2d = self.layer2db2(x_a_2d)
        x_a_2d = self.layer2db3(x_a_2d)
        x_a_2d = self.layer2db4(x_a_2d)
        x_a1_2d = self.adaptiveavgpool2d(x_a_2d)
        x_a2_2d = self.adaptivemaxpool2d(x_a_2d)
        x_a_2d = torch.cat((x_a1_2d, x_a2_2d), dim=1)        # 1024
        x_a_2d = x_a_2d.view(x_a_2d.size(0), -1)

        x_2d = torch.cat((x_m_2d, x_a_2d), dim=1)            # 2048

        p_loss_2d = F.kl_div(F.log_softmax(x_m_2d, dim=-1), F.softmax(x_a_2d, dim=-1), reduction='mean')
        q_loss_2d = F.kl_div(F.log_softmax(x_a_2d, dim=-1), F.softmax(x_m_2d, dim=-1), reduction='mean')
        loss_kl_2d = 0.5 * (p_loss_2d + q_loss_2d)
        loss_kl = loss_kl_1d + loss_kl_2d

        x = torch.cat((x_1d, x_2d), dim=1)                   # 4096
        x = x.view(x.size(0), -1)

        x = self.relu2d(x)
        x = self.fc4(x)     # 1024
        x_f = x.clone()
        x = self.relu2d(x)
        x = self.fc5(x)     # 12
        x_f = torch.cat((x_f, x), dim=1)

        if args.get_feature:
            return x_f

        return x, loss_kl


# 在模型中只使用1维卷积
class ResNet1d(nn.Module):
    def __init__(self, block_a, block_b, layers, num_classes, input_channels, out_channels=64):
        super(ResNet1d, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(input_channels, self.out_channels, kernel_size=50, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.layera1 = self._make_layer(block_a, 64, layers[0])
        self.layera2 = self._make_layer(block_a, 128, layers[1], stride=2)
        self.layera3 = self._make_layer(block_a, 256, layers[2], stride=2)
        self.layera4 = self._make_layer(block_a, 512, layers[3], stride=2)

        self.out_channels = out_channels  # 将layerb1的输入维度恢复为64维，否则会成为512维
        self.layerb1 = self._make_layer(block_b, 64, layers[0])
        self.layerb2 = self._make_layer(block_b, 128, layers[1], stride=2)
        self.layerb3 = self._make_layer(block_b, 256, layers[2], stride=2)
        self.layerb4 = self._make_layer(block_b, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)

        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc1 = nn.Linear(512 * block_b.expansion * 2, 128)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.out_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x_m = x
        x_a = x

        x_m = self.conv1(x_m)
        x_m = self.bn1(x_m)
        x_m = self.relu(x_m)
        x_m = self.conv2(x_m)
        x_m = self.bn2(x_m)
        x_m = self.relu(x_m)
        x_m = self.maxpool(x_m)
        x_m = self.layera1(x_m)
        x_m = self.layera2(x_m)
        x_m = self.layera3(x_m)
        x_m = self.layera4(x_m)
        x_m1 = self.adaptiveavgpool(x_m)
        x_m2 = self.adaptivemaxpool(x_m)
        x_m = torch.cat((x_m1, x_m2), dim=1)
        x_m = x_m.view(x_m.size(0), -1)

        x_a = self.conv1(x_a)
        x_a = self.bn1(x_a)
        x_a = self.relu(x_a)
        x_a = self.avgpool(x_a)         # todo
        x_a = self.layerb1(x_a)
        x_a = self.layerb2(x_a)
        x_a = self.layerb3(x_a)
        x_a = self.layerb4(x_a)
        x_a1 = self.adaptiveavgpool(x_a)
        x_a2 = self.adaptivemaxpool(x_a)
        x_a = torch.cat((x_a1, x_a2), dim=1)        # 1024
        x_a = x_a.view(x_a.size(0), -1)

        x = torch.cat((x_m, x_a), dim=1)            # 2048
        x = x.view(x.size(0), -1)

        x = self.relu(x)
        x = self.fc2(x)     # 512
        x_f = x.clone()
        x = self.relu(x)
        x = self.fc3(x)     # 12
        x_f = torch.cat((x_f, x), dim=1)

        if args.get_feature:
            return x_f

        p_loss = F.kl_div(F.log_softmax(x_m, dim=-1), F.softmax(x_a, dim=-1), reduction='mean')
        q_loss = F.kl_div(F.log_softmax(x_a, dim=-1), F.softmax(x_m, dim=-1), reduction='mean')
        loss_kl = 0.5 * (p_loss + q_loss)

        return x, loss_kl


# 在模型中只使用1维卷积,第二层为256维度
class ResNet1dMini(nn.Module):
    def __init__(self, block_a, block_b, layers, num_classes, input_channels, out_channels=64):
        super(ResNet1dMini, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(input_channels, self.out_channels, kernel_size=50, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.layera1 = self._make_layer(block_a, 64, layers[0])
        self.layera2 = self._make_layer(block_a, 128, layers[1], stride=2)
        self.layera3 = self._make_layer(block_a, 256, layers[2], stride=2)
        self.layera4 = self._make_layer(block_a, 512, layers[3], stride=2)

        self.out_channels = out_channels  # 将layerb1的输入维度恢复为64维，否则会成为512维
        self.layerb1 = self._make_layer(block_b, 64, layers[0])
        self.layerb2 = self._make_layer(block_b, 128, layers[1], stride=2)
        self.layerb3 = self._make_layer(block_b, 256, layers[2], stride=2)
        self.layerb4 = self._make_layer(block_b, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)

        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc1 = nn.Linear(512 * block_b.expansion * 2, 128)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.out_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x_m = x
        x_a = x

        x_m = self.conv1(x_m)
        x_m = self.bn1(x_m)
        x_m = self.relu(x_m)
        x_m = self.conv2(x_m)
        x_m = self.bn2(x_m)
        x_m = self.relu(x_m)
        x_m = self.maxpool(x_m)
        x_m = self.layera1(x_m)
        x_m = self.layera2(x_m)
        x_m = self.layera3(x_m)
        x_m = self.layera4(x_m)
        x_m1 = self.adaptiveavgpool(x_m)
        x_m2 = self.adaptivemaxpool(x_m)
        x_m = torch.cat((x_m1, x_m2), dim=1)
        x_m = x_m.view(x_m.size(0), -1)

        x_a = self.conv1(x_a)
        x_a = self.bn1(x_a)
        x_a = self.relu(x_a)
        x_a = self.avgpool(x_a)         # todo
        x_a = self.layerb1(x_a)
        x_a = self.layerb2(x_a)
        x_a = self.layerb3(x_a)
        x_a = self.layerb4(x_a)
        x_a1 = self.adaptiveavgpool(x_a)
        x_a2 = self.adaptivemaxpool(x_a)
        x_a = torch.cat((x_a1, x_a2), dim=1)        # 1024
        x_a = x_a.view(x_a.size(0), -1)

        x = torch.cat((x_m, x_a), dim=1)            # 2048
        x = x.view(x.size(0), -1)

        x = self.relu(x)
        x = self.fc2(x)     # 512
        x_f = x.clone()
        x = self.relu(x)
        x = self.fc3(x)     # 12
        x_f = torch.cat((x_f, x), dim=1)

        if args.get_feature:
            return x_f

        p_loss = F.kl_div(F.log_softmax(x_m, dim=-1), F.softmax(x_a, dim=-1), reduction='mean')
        q_loss = F.kl_div(F.log_softmax(x_a, dim=-1), F.softmax(x_m, dim=-1), reduction='mean')
        loss_kl = 0.5 * (p_loss + q_loss)

        return x, loss_kl


# 在模型中只使用2维卷积
class ResNet2d(nn.Module):
    def __init__(self, block_a, block_b, layers, num_classes, input_channels, out_channels=64):
        super(ResNet2d, self).__init__()
        self.out_channels = out_channels
        self.conv2d1 = nn.Conv2d(input_channels, self.out_channels, kernel_size=(1, 50), stride=(1, 2), padding=(0, 7), bias=False)
        self.bn2d1 = nn.BatchNorm2d(out_channels)
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.relu2d = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.layer2da1 = self._make_layer_2d(block_a, 64, layers[0])
        self.layer2da2 = self._make_layer_2d(block_a, 128, layers[1], stride=(1, 2))
        self.layer2da3 = self._make_layer_2d(block_a, 256, layers[2], stride=(1, 2))
        self.layer2da4 = self._make_layer_2d(block_a, 512, layers[3], stride=(1, 2))

        self.out_channels = out_channels
        self.layer2db1 = self._make_layer_2d(block_b, 64, layers[0])
        self.layer2db2 = self._make_layer_2d(block_b, 128, layers[1], stride=(1, 2))
        self.layer2db3 = self._make_layer_2d(block_b, 256, layers[2], stride=(1, 2))
        self.layer2db4 = self._make_layer_2d(block_b, 512, layers[3], stride=(1, 2))
        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptivemaxpool2d = nn.AdaptiveMaxPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc1 = nn.Linear(512 * block_a.expansion * 2, 128)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer_2d(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != (1, 1) or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, planes * block.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_m = x.unsqueeze(2)
        x_a = x.unsqueeze(2)

        x_m = self.conv2d1(x_m)
        x_m = self.bn2d1(x_m)
        x_m = self.relu2d(x_m)
        x_m = self.conv2d2(x_m)
        x_m = self.bn2d2(x_m)
        x_m = self.relu2d(x_m)
        x_m = self.maxpool2d(x_m)

        x_m = self.layer2da1(x_m)
        x_m = self.layer2da2(x_m)
        x_m = self.layer2da3(x_m)
        x_m = self.layer2da4(x_m)
        x_m1 = self.adaptiveavgpool2d(x_m)
        x_m2 = self.adaptivemaxpool2d(x_m)
        x_m = torch.cat((x_m1, x_m2), dim=1)
        x_m = x_m.view(x_m.size(0), -1)

        x_a = self.conv2d1(x_a)
        x_a = self.bn2d1(x_a)
        x_a = self.relu2d(x_a)
        x_a = self.conv2d2(x_a)
        x_a = self.bn2d2(x_a)
        x_a = self.relu2d(x_a)
        x_a = self.avgpool2d(x_a)

        x_a = self.layer2db1(x_a)
        x_a = self.layer2db2(x_a)
        x_a = self.layer2db3(x_a)
        x_a = self.layer2db4(x_a)
        x_a1 = self.adaptiveavgpool2d(x_a)
        x_a2 = self.adaptivemaxpool2d(x_a)
        x_a = torch.cat((x_a1, x_a2), dim=1)        # 1024
        x_a = x_a.view(x_a.size(0), -1)

        x = torch.cat((x_m, x_a), dim=1)            # 2048
        x = x.view(x.size(0), -1)

        x = self.relu2d(x)
        x = self.fc2(x)     # 512
        x_f = x.clone()
        x = self.relu2d(x)
        x = self.fc3(x)     # 12
        x_f = torch.cat((x_f, x), dim=1)

        if args.get_feature:
            return x_f

        p_loss = F.kl_div(F.log_softmax(x_m, dim=-1), F.softmax(x_a, dim=-1), reduction='mean')
        q_loss = F.kl_div(F.log_softmax(x_a, dim=-1), F.softmax(x_m, dim=-1), reduction='mean')
        loss_kl = 0.5 * (p_loss + q_loss)

        return x, loss_kl


# 在模型中只使用2维卷积,倒数第二层为256维度
class ResNet2dMini(nn.Module):
    def __init__(self, block_a, block_b, layers, num_classes, input_channels, out_channels=64):
        super(ResNet2dMini, self).__init__()
        self.out_channels = out_channels
        self.conv2d1 = nn.Conv2d(input_channels, self.out_channels, kernel_size=(1, 50), stride=(1, 2), padding=(0, 7), bias=False)
        self.bn2d1 = nn.BatchNorm2d(out_channels)
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.relu2d = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.layer2da1 = self._make_layer_2d(block_a, 64, layers[0])
        self.layer2da2 = self._make_layer_2d(block_a, 128, layers[1], stride=(1, 2))
        self.layer2da3 = self._make_layer_2d(block_a, 256, layers[2], stride=(1, 2))
        self.layer2da4 = self._make_layer_2d(block_a, 512, layers[3], stride=(1, 2))

        self.out_channels = out_channels
        self.layer2db1 = self._make_layer_2d(block_b, 64, layers[0])
        self.layer2db2 = self._make_layer_2d(block_b, 128, layers[1], stride=(1, 2))
        self.layer2db3 = self._make_layer_2d(block_b, 256, layers[2], stride=(1, 2))
        self.layer2db4 = self._make_layer_2d(block_b, 512, layers[3], stride=(1, 2))
        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptivemaxpool2d = nn.AdaptiveMaxPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.fc1 = nn.Linear(512 * block_a.expansion * 2, 128)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer_2d(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != (1, 1) or self.out_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, planes * block.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, planes, stride, downsample))
        self.out_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_m = x.unsqueeze(2)
        x_a = x.unsqueeze(2)

        x_m = self.conv2d1(x_m)
        x_m = self.bn2d1(x_m)
        x_m = self.relu2d(x_m)
        x_m = self.conv2d2(x_m)
        x_m = self.bn2d2(x_m)
        x_m = self.relu2d(x_m)
        x_m = self.maxpool2d(x_m)

        x_m = self.layer2da1(x_m)
        x_m = self.layer2da2(x_m)
        x_m = self.layer2da3(x_m)
        x_m = self.layer2da4(x_m)
        x_m1 = self.adaptiveavgpool2d(x_m)
        x_m2 = self.adaptivemaxpool2d(x_m)
        x_m = torch.cat((x_m1, x_m2), dim=1)
        x_m = x_m.view(x_m.size(0), -1)

        x_a = self.conv2d1(x_a)
        x_a = self.bn2d1(x_a)
        x_a = self.relu2d(x_a)
        x_a = self.conv2d2(x_a)
        x_a = self.bn2d2(x_a)
        x_a = self.relu2d(x_a)
        x_a = self.avgpool2d(x_a)

        x_a = self.layer2db1(x_a)
        x_a = self.layer2db2(x_a)
        x_a = self.layer2db3(x_a)
        x_a = self.layer2db4(x_a)
        x_a1 = self.adaptiveavgpool2d(x_a)
        x_a2 = self.adaptivemaxpool2d(x_a)
        x_a = torch.cat((x_a1, x_a2), dim=1)        # 1024
        x_a = x_a.view(x_a.size(0), -1)

        x = torch.cat((x_m, x_a), dim=1)            # 2048
        x = x.view(x.size(0), -1)

        x = self.relu2d(x)
        x = self.fc2(x)     # 512
        x_f = x.clone()
        x = self.relu2d(x)
        x = self.fc3(x)     # 12
        x_f = torch.cat((x_f, x), dim=1)

        if args.get_feature:
            return x_f

        p_loss = F.kl_div(F.log_softmax(x_m, dim=-1), F.softmax(x_a, dim=-1), reduction='mean')
        q_loss = F.kl_div(F.log_softmax(x_a, dim=-1), F.softmax(x_m, dim=-1), reduction='mean')
        loss_kl = 0.5 * (p_loss + q_loss)

        return x, loss_kl


def resnet181d3_5(**kwargs):
    logger.info("using resnet18 1d model")
    model = ResNet1d(BasicBlock1d3_3, BasicBlock1d5_5, [2, 2, 2, 2], **kwargs)
    return model

def resnet341d3_5(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1dMini(BasicBlock1d3_3, BasicBlock1d5_5, [3, 4, 6, 3], **kwargs)
    return model

def resnet341d3_7(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1d(BasicBlock1d3_3, BasicBlock1d7_7, [3, 4, 6, 3], **kwargs)
    return model

def resnet341d3_15(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1d(BasicBlock1d3_3, BasicBlock1d15_15, [3, 4, 6, 3], **kwargs)
    return model

def resnet341d5_7(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1d(BasicBlock1d5_5, BasicBlock1d7_7, [3, 4, 6, 3], **kwargs)
    return model

def resnet341d5_15(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1d(BasicBlock1d5_5, BasicBlock1d15_15, [3, 4, 6, 3], **kwargs)
    return model

def resnet341d7_15(**kwargs):
    logger.info("using resnet34 1d model")
    model = ResNet1d(BasicBlock1d7_7, BasicBlock1d15_15, [3, 4, 6, 3], **kwargs)
    return model



def resnet182d3_5(**kwargs):
    logger.info("using resnet18 2d model")
    model = ResNet2dMini(BasicBlock2d3_3, BasicBlock2d5_5, [2, 2, 2, 2], **kwargs)
    return model

def resnet342d3_5(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d3_3, BasicBlock2d5_5, [3, 4, 6, 3], **kwargs)
    return model

def resnet342d3_7(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d3_3, BasicBlock2d7_7, [3, 4, 6, 3], **kwargs)
    return model

def resnet342d3_15(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d3_3, BasicBlock2d15_15, [3, 4, 6, 3], **kwargs)
    return model

def resnet342d5_7(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d5_5, BasicBlock2d7_7, [3, 4, 6, 3], **kwargs)
    return model

def resnet342d5_15(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d5_5, BasicBlock2d15_15, [3, 4, 6, 3], **kwargs)
    return model

def resnet342d7_15(**kwargs):
    logger.info("using resnet34 2d model")
    model = ResNet2d(BasicBlock2d7_7, BasicBlock2d15_15, [3, 4, 6, 3], **kwargs)
    return model