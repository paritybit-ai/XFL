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
        import torch.optim.lr_scheduler as torch_lr_scheduler
    if k == "paddle":
        import paddle.optimizer.lr as paddle_lr_scheduler


def get_lr_scheduler(name: str, framework:str="torch"):
    scheduler = None
    if framework == "torch":
        if name in dir(torch_lr_scheduler):
            scheduler = getattr(torch_lr_scheduler, name)
    elif framework == "paddle":
        if name in dir(paddle_lr_scheduler):
            scheduler = getattr(paddle_lr_scheduler, name)
    elif name in dir(sys.modules[__name__]):
        scheduler = getattr(sys.modules[__name__], name)
    elif name in xregister.registered_object:
        scheduler = xregister(name)
    else:
        raise ValueError(f"Scheduler {name} not support.")
    return scheduler