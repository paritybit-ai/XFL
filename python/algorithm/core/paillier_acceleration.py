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


from typing import List

import numpy as np


def embed(p_list: List[np.ndarray], interval: int = (1 << 128), precision: int = 64):
    def _embed(_p_list):
        x = int(_p_list[0] * (1 << precision))
        for i in range(len(_p_list) - 1):
            x = x * interval + int(_p_list[i+1] * (1 << precision))
        return x
    
    out = [0] * len(p_list[0])
    for i in range(len(p_list[0])):
        _p_list = [p_list[j][i] for j in range(len(p_list))]
        out[i] = _embed(_p_list)
    return np.array(out)
    

def umbed(a: np.ndarray, num: int, interval: int = (1 << 128), precison: int = 64) -> List[list]:
    def _umbed(x):
        res = [0] * num
        # a, b = divmod(x, interval)
        b = x % interval
        if abs(b) > interval // 2:
            b = b - interval
        a = (x - b) // interval
        
        res[-1] = b / (1 << precison)
        for i in range(num -1):
            # y, b = divmod(a, interval)
            b = a % interval
            if abs(b) > interval // 2:
                b = b - interval
            a = (a - b) // interval
            res[-i-2] = b / (1 << precison)
        return np.array(res).astype(np.float32)
    
    out = [[0] * len(a) for i in range(num)]
    for i in range(len(a)):
        temp = _umbed(a[i])
        for j in range(num):
            out[j][i] = temp[j]
    return out
