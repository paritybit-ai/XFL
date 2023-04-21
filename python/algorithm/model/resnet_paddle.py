# Copyright 2022 The XFL Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This model contains a PaddlePaddle implementation of the paper "Deep Residual Learning for Image Recognition."[1]
# [1]He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

from collections import OrderedDict
import paddle
from paddle import nn 

class ConvBlock(nn.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.stem = nn.Sequential(
            ("conv1", nn.Conv2D(in_channels, out_channels,
             kernel_size=1, stride=1, padding=0)),
            ("batch_norm1", nn.BatchNorm2D(out_channels)), 
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2D(out_channels, out_channels,
             kernel_size=3, stride=stride, padding=1)),
            ("batch_norm2", nn.BatchNorm2D(out_channels)),
            ("relu2", nn.ReLU()),
            ("conv3", nn.Conv2D(out_channels, out_channels *
             self.expansion, kernel_size=1, stride=1, padding=0)),
            ("batch_norm3", nn.BatchNorm2D(out_channels*self.expansion))
        )

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.stem(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class Resnet(nn.Layer):
    def __init__(self, ResBlock, block_list, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            ("conv1", nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)),
            ("batch_norm1", nn.BatchNorm2D(64)),
            ("relu", nn.ReLU())
        )
        self.max_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layers1 = self._make_layers(
            ResBlock, block_list[0], inplanes=64, outplanes=64, stride=1)
        self.layers2 = self._make_layers(
            ResBlock, block_list[1], inplanes=256, outplanes=128, stride=2)
        self.layers3 = self._make_layers(
            ResBlock, block_list[2], inplanes=512, outplanes=256, stride=2)
        self.layers4 = self._make_layers(
            ResBlock, block_list[3], inplanes=1024, outplanes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.max_pool(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fc(x)
        return x

    def _make_layers(self, ResBlock, blocks, inplanes, outplanes, stride=1):
        layers =[]
        downsample = None
        if stride != 1 or inplanes != outplanes*ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(inplanes, outplanes*ResBlock.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2D(outplanes*ResBlock.expansion)
            )

        layers.append(ResBlock(inplanes, outplanes,
                      downsample=downsample, stride=stride))

        for i in range(1, blocks):
            layers.append(ResBlock(outplanes*ResBlock.expansion, outplanes))

        return nn.Sequential(*layers)


def ResNet(num_classes, layers):
    if layers == "unit_test":
        return Resnet(ConvBlock, [2, 2, 2, 2], num_classes) 
    elif layers == 50:
        return Resnet(ConvBlock, [3, 4, 6, 3], num_classes) 
    elif layers == 101:
        return Resnet(ConvBlock, [3, 4, 23, 3], num_classes)
    elif layers == 152:
        return Resnet(ConvBlock, [3, 8, 36, 3], num_classes)
    else:
        raise NotImplementedError("Only support ResNet50, ResNet101, ResNet152 currently, please change layers")

