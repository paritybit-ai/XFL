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


from typing import Optional, Union

import numpy as np
import pandas as pd


def cal_cat_rank(sum_grad: Union[pd.Series, np.ndarray],
                 sum_hess: Union[pd.Series, np.ndarray],
                 cat_smooth: float) -> Union[pd.Series, np.ndarray]:
    return sum_grad / (sum_hess + cat_smooth)


def cal_gain(cum_grad: np.ndarray,
             cum_hess: np.ndarray,
             lambda_: float) -> np.ndarray:
    if len(cum_grad) <= 1:
        return np.array([-float('inf')], dtype=np.float32)

    grad_left = cum_grad[:-1]
    grad_right = cum_grad[-1] - grad_left

    hess_left = cum_hess[:-1]
    hess_right = cum_hess[-1] - hess_left

    base_score = np.square(cum_grad[-1]) / (cum_hess[-1] + lambda_)
    gain = np.square(grad_left) / (hess_left + lambda_) + \
                np.square(grad_right) / (hess_right + lambda_) - \
                    base_score
    return gain


# def cal_gain(cum_grad: np.ndarray,
#              cum_hess: np.ndarray,
#              lambda_: float,
#              grad_missing: Optional[float] = None,
#              hess_missing: Optional[float] = None) -> np.ndarray:
#     if len(cum_grad) <= 1:
#         return np.array([-float('inf')], dtype=np.float32)

#     grad_left = cum_grad[:-1]
#     grad_right = cum_grad[-1] - grad_left

#     hess_left = cum_hess[:-1]
#     hess_right = cum_hess[-1] - hess_left

#     if grad_missing is None or hess_missing is None or (grad_missing == 0 and hess_missing == 0):
#         base_score = np.square(cum_grad[-1]) / (cum_hess[-1] + lambda_)
#         gain = np.square(grad_left) / (hess_left + lambda_) + \
#                     np.square(grad_right) / (hess_right + lambda_) - \
#                         base_score
#     else:
#         base_score = np.square(cum_grad[-1] + grad_missing) / (cum_hess[-1] + hess_missing + lambda_)
#         gain_missing_on_left = np.square(grad_left + grad_missing) / (hess_left + hess_missing + lambda_) + \
#                                     np.square(grad_right) / (hess_right + lambda_) - \
#                                         base_score
#         gain_missing_on_right = np.square(grad_left) / (hess_left + lambda_) + \
#                                     np.square(grad_right + grad_missing) / (hess_right + hess_missing + lambda_) - \
#                                         base_score
#         gain = np.concatenate([gain_missing_on_left, gain_missing_on_right])
#     return gain
    
    
def cal_weight(sum_grad: float, 
               sum_hess: float, 
               lambda_: float) -> float:
    return -sum_grad / (sum_hess + lambda_)
    
    
class BestSplitInfo(object):
    def __init__(self,
                 gain: float = -float('inf'),
                 feature_ower: str = '',
                 feature_index: int = 0, 
                 is_category: bool = False,
                 split_point: Optional[float] = None,
                 left_cat: Optional[list] = None,
                 missing_value_on_left: bool = None,
                 left_sample_index: Optional[np.ndarray] = None,
                 right_sample_index: Optional[np.ndarray] = None,
                 left_bin_weight: float = 0,
                 right_bin_weight: float = 0,
                 num_left_bin: Optional[int] = None,
                 num_right_bin: Optional[int] = None,
                 max_gain_index: Optional[int] = None): 
        self.gain = gain
        self.feature_owner = feature_ower
        self.feature_idx = feature_index
        self.is_category = is_category
        self.split_point = split_point
        self.left_cat = left_cat
        self.missing_value_on_left = missing_value_on_left
        self.left_sample_index = left_sample_index
        self.right_sample_index = right_sample_index
        self.left_bin_weight = left_bin_weight
        self.right_bin_weight = right_bin_weight
        self.num_left_bin = num_left_bin
        self.num_right_bin = num_right_bin
        self.max_gain_index = max_gain_index
