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

# This model contains a PyTorch implementation of the paper "Densely Connected Convolutional Networks."[1]
# [1]Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2016). Densely connected convolutional networks. arXiv preprint arXiv:1608.06993.

from collections import OrderedDict
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, growth_rate, drop_out=0.0):
        super().__init__()
        self.conv_block1 = nn.Sequential(OrderedDict([
            ("batch_norm1", nn.BatchNorm2d(in_planes, track_running_stats=True)), # setting track_running_stats as False 
            ("relu1", nn.ReLU()),
            ("conv1", nn.Conv2d(in_planes, self.expansion*growth_rate, kernel_size=1, stride=1, bias=False))
        ]))
           
        self.conv_block2 = nn.Sequential(OrderedDict([
            ("batch_norm2", nn.BatchNorm2d(self.expansion*growth_rate, track_running_stats=True)),
            ("relu2", nn.ReLU()),
            ("conv2", nn.Conv2d(self.expansion*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.drop_out = drop_out

    def forward(self, x):
        out = self.conv_block1(x)
        if self.drop_out:
            out = F.dropout(out, p=self.drop_out, training=self.training)
        out = self.conv_block2(out)
        if self.drop_out:
            out = F.dropout(out, p=self.drop_out, training=self.training)
        
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_out=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop_out = drop_out

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_out:
            out = F.dropout(out, p=self.drop_out, training=self.training)
        return F.avg_pool2d(out, 2)


class Densenet(nn.Module):
    def __init__(self, block, block_list, num_classes, growth_rate=12, reduction=0.5, drop_out=0.0):
        super().__init__()
        self.growth_rate = growth_rate
        self.drop_out = drop_out
        in_planes = 2 * growth_rate
        self.conv = nn.Conv2d(3, in_planes, kernel_size=3, padding=1, bias=False)
        self.dense_layer1 = self._make_layers(block, block_list[0], in_planes)
        in_planes += block_list[0]*growth_rate
        self.transition1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), drop_out=drop_out)
        in_planes = int(math.floor(in_planes*reduction))

        self.dense_layer2 = self._make_layers(block, block_list[1], in_planes)
        in_planes += block_list[1]*growth_rate
        self.transition2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), drop_out=drop_out)
        in_planes = int(math.floor(in_planes*reduction))

        self.dense_layer3 = self._make_layers(block, block_list[2], in_planes)
        in_planes += block_list[2]*growth_rate
        self.transition3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), drop_out=drop_out)
        in_planes = int(math.floor(in_planes*reduction))

        self.dense_layer4 = self._make_layers(block, block_list[3], in_planes)
        in_planes += block_list[3]*growth_rate

        self.batchnorm = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.transition1(self.dense_layer1(x))
        x = self.transition2(self.dense_layer2(x))
        x = self.transition3(self.dense_layer3(x))
        x = self.dense_layer4(x)
        x = self.relu(self.batchnorm(x))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layers(self, block, blocks, in_planes):
        layers = []
        for i in range(blocks):
            layers.append(block(in_planes, self.growth_rate, drop_out=self.drop_out))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)



def DenseNet(num_classes, layers):
    if layers == 121:
        return Densenet(BottleNeckBlock, [6,12,24,16],  num_classes, growth_rate=32) 
    elif layers == 169:
        return Densenet(BottleNeckBlock, [6,12,32,32], num_classes, growth_rate=32)
    elif layers == 201:
        return Densenet(BottleNeckBlock, [6,12,48,32], num_classes, growth_rate=32)
    elif layers == 264:
        return Densenet(BottleNeckBlock, [6,12,64,48], num_classes, growth_rate=32)
    elif layers == 'unit_test':
        return Densenet(BottleNeckBlock, [2,2,2,2], num_classes, growth_rate=8)
    else:
        raise NotImplementedError("Only support DenseNet121, DenseNet169, DenseNet201, DenseNet264 currently, please change layers")

