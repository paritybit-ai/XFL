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
import paddle.nn as paddle_nn
from common.xregister import xregister


def get_lossfunc(name: str):
    if name in dir(paddle_nn):
        loss_func = getattr(paddle_nn, name)
    elif name in dir(sys.modules[__name__]):
        loss_func = getattr(sys.modules[__name__], name)
    elif name in xregister.registered_object:
        loss_func = xregister(name)
    else:
        raise ValueError(f"Loss function {name} is not supported in paddlepaddle.")
    return loss_func
    