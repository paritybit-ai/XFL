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

# This model contains a PyTorch implementation of the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition."[1]
# [1]Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

import torch
import torch.nn as nn


layers_cfg = {
    'VGG11': [64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'],
    'VGG13': [64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'],
    'VGG16': [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512, 512, 512, 'max'],
    'VGG19': [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512, 512, 512, 512, 'max'],
}


class Vgg(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super().__init__()
        self.stem = self._make_layers(layers_cfg[vgg_name])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layers(self, layers_cfg):
        layers = []
        in_planes = 3
        for outplanes in layers_cfg:
            if outplanes == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_planes, outplanes, kernel_size=3, padding=1),
                           nn.BatchNorm2d(outplanes),
                           nn.ReLU(inplace=True)])
                in_planes = outplanes
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)


def VGG(num_classes, layers):
    if layers == 11:
        return Vgg("VGG11", num_classes)
    elif layers == 13:
        return Vgg("VGG13", num_classes)
    elif layers == 16:
        return Vgg("VGG16", num_classes)
    elif layers == 19:
        return Vgg("VGG19", num_classes)
    else:
        raise NotImplementedError("Only support VGG11, VGG13, VGG16, VGG!9 currently, please change layers")