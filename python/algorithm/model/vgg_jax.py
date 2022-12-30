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

import jax.numpy as jnp
import flax.linen as nn


layers_cfg = {
    'VGG11': [64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'],
    'VGG13': [64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'],
    'VGG16': [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512, 512, 512, 'max'],
    'VGG19': [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512, 512, 512, 512, 'max'],
    'unit_test': [64, 'max', 128, 'max', 256,  'max', 512, 'max']
}


class VggJax(nn.Module):
    vgg_name: str
    num_classes: int
    
    @nn.compact
    def __call__(self, x, train=True):
        
        def adaptive_avg_pool(x):
            return nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=(1,1))
        
        def seq_max_pool(x):
            return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        
        layers = []
        for outplanes in layers_cfg[self.vgg_name]:
            if outplanes == 'max':
                layers.append(seq_max_pool)
            else:
                layers.extend([
                    nn.Conv(features=outplanes, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=jnp.float32),
                    nn.relu
                ])
        layers.append(adaptive_avg_pool)
        
        model = nn.Sequential(layers)
        fc = nn.Dense(self.num_classes)
        
        x = model(x)
        x = x.reshape((x.shape[0], -1))
        x = fc(x)
        return x


def vggjax(num_classes, layers):
    if layers == 11:
        return VggJax("VGG11", num_classes)
    elif layers == 13:
        return VggJax("VGG13", num_classes)
    elif layers == 16:
        return VggJax("VGG16", num_classes)
    elif layers == 19:
        return VggJax("VGG19", num_classes)
    elif layers == "unit_test":
        return VggJax("unit_test", num_classes)
    else:
        raise NotImplementedError("Only support VGG11, VGG13, VGG16, VGG19 currently, please change layers")