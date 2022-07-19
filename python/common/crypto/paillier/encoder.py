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


import math
import sys
from typing import Optional, Union

import numpy as np

from .context import PaillierContext
from .utils import MPZ, mul


class PaillierEncoder(object):
    _MANT_DIG = sys.float_info.mant_dig
    
    @classmethod
    def cal_exponent(cls,
                     data: Union[int, float, np.ndarray, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64],
                     precision: Optional[int] = None) -> Union[int, np.ndarray]:
        """Precison are expected to be an non-negative integer.
        """
        if precision is None:
            if isinstance(data, np.ndarray):
                exponent = np.frexp(data)[1] - cls._MANT_DIG
            elif isinstance(data, (np.int32, np.int64, int, np.int16)):
                exponent = 0
            elif isinstance(data, (np.float32, np.float64, float, np.float16,)):
                exponent = math.frexp(data)[1] - cls._MANT_DIG
            else:
                raise TypeError(f"Precision type {type(precision)} not supported.")
        else:
            exponent = -math.ceil(math.log2(10) * precision)
        return exponent
    
    @classmethod
    def encode_single(cls, 
                      context: PaillierContext,
                      data: Union[int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64],
                      exponent: int) -> MPZ:
        out = round(data * (1 << -exponent)) % context.n
        return out
    
    @classmethod
    def decode_single(cls, context: PaillierContext, data: MPZ, exponent: int) -> float:
        if data >= context.min_value_for_negative:
            data -= context.n
        elif data > context.max_value_for_positive:
            raise OverflowError("Overflow detected during decoding encrypted number.")
        
        out = mul(data, pow(2, exponent))
        return out
