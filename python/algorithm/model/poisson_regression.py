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


import torch
import torch.nn as nn


class PoissonRegression(nn.Module):
    def __init__(self, input_dim: int, bias: bool = False):
        super(PoissonRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x):
        return torch.exp(self.linear(x))
