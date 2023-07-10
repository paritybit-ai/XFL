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


import sys
import torch.nn as torch_nn
from torch.nn import Module
import torch
from common.xregister import xregister


def get_lossfunc(name: str):
    if name in dir(torch_nn):
        lossfunc = getattr(torch_nn, name)
    elif name in dir(sys.modules[__name__]):
        lossfunc = getattr(sys.modules[__name__], name)
    elif name in xregister.registered_object:
        lossfunc = xregister(name)
    else:
        raise ValueError(f"Loss function {name} is not supported in torch.")
    return lossfunc


class MapeLoss(Module):
    def __init__(self):
        super(MapeLoss, self).__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels != 0)
        distance = torch.abs(preds - labels) / torch.abs(labels)
        return torch.mean(distance[mask])
