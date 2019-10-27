import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # For add x to F(x), reduce size of x
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        first_ch = 16
        self.conv = nn.Conv2d(3, first_ch, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(first_ch)
        l1_out_ch = 16
        self.layer1 = self.make_layer(
            first_ch, l1_out_ch, block, num_blocks[0], stride=1)
        l2_out_ch = 32
        self.layer2 = self.make_layer(
            l1_out_ch, l2_out_ch, block, num_blocks[1], stride=2)
        l3_out_ch = 64
        self.layer3 = self.make_layer(
            l2_out_ch, l3_out_ch, block, num_blocks[2], stride=2)
        l4_out_ch = 128
        self.layer4 = self.make_layer(
            l3_out_ch, l4_out_ch, block, num_blocks[3], stride=2)
        self.linear = nn.Linear(l4_out_ch * 16 * 9, num_classes)

    def make_layer(self, in_channels, out_channels, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        prev_out = in_channels
        for stride in strides:
            layers.append(block(prev_out, out_channels, stride))
            prev_out = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)  # flatten
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(block=ResidualBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
