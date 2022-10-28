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
from common.xregister import xregister

module_list = list(sys.modules.keys())
for k in module_list:
    if k == "torch":
        import torch.optim as torch_optim
    if k == "tensorflow":
        import tensorflow.keras.optimizers as tf_optim
    if k == "paddle":
        import paddle.optimizer as pd_optim


def get_optimizer(name: str, framework: str="torch"):
    optim = None
    if framework == "torch":
        if name in dir(torch_optim):
            optim = getattr(torch_optim, name)
    elif framework == "tf":
        if name in dir(tf_optim):
            optim = getattr(tf_optim, name)
    elif framework == "paddle":
        if name in dir(pd_optim):
            optim = getattr(pd_optim, name)
    elif name in dir(sys.modules[__name__]):
        optim = getattr(sys.modules[__name__], name)
    elif name in xregister.registered_object:
        optim = xregister(name)
    else:
        raise ValueError(f"Optimizer {name} not support.")
    return optim

