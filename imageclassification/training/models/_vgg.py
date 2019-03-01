import torch.nn as nn
import torch.nn.functional as F

import math


class VGGBNDrop(nn.Module):
    """
    https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua

    """
    def __init__(self, num_classes, init_weights=True):
        super(VGGBNDrop, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # vgg:add(MaxPooling(2,2,2,2):ceil())
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # classifier:add(nn.Dropout(0.5))
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # classifier:add(nn.BatchNormalization(512))
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # v.kW*v.kH*v.nOutputPlane
                m.weight.data.normal_(0, math.sqrt(2. / n))  # v.weight:normal(0,math.sqrt(2/n))
                if m.bias is not None:
                    m.bias.data.zero_()  # v.bias:zero()
            if isinstance(m, nn.Linear):  # Antti mod
                m.bias.data.zero_()  # v.bias: zero()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    """
    https://github.com/szagoruyko/wide-residual-networks/blob/master/models/vgg.lua

    """
    def __init__(self, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, 1e-3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # vgg:add(MaxPooling(2,2,2,2):ceil())

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(inplace=True),

            #nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)  # model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        https://github.com/szagoruyko/wide-residual-networks/blob/master/models/utils.lua#L3-L9

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # v.kW*v.kH*v.nOutputPlane
                m.weight.data.normal_(0, math.sqrt(2. / n))  # v.weight:normal(0,math.sqrt(2/n))
                if m.bias is not None:
                    m.bias.data.zero_()  # v.bias:zero()
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()  # v.bias: zero()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)  # Antti mod, applies a 2D adaptive average pooling over an input signal composed of several input planes
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
