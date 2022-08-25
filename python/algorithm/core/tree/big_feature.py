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


from typing import Optional

import numpy as np
import pandas as pd


class Feature(object):
    """
    Process column data.
    """
    def __init__(self, data: pd.DataFrame, feature_columns: list):
        self.data = data
        self.feature_columns = feature_columns

    @classmethod
    def create(cls,
               values: pd.DataFrame,
               sample_index: Optional[np.ndarray] = None,
               grad: Optional[np.ndarray] = None,
               hess: Optional[np.ndarray] = None,
               grad_hess: Optional[np.ndarray] = None): 
        
        values.reset_index(drop=True, inplace=True)
        
        if sample_index is None:
            sample_index = range(values.shape[0])
        
        if grad_hess is not None:
            data = pd.concat([pd.DataFrame(sample_index, columns=['xfl_id']),
                              pd.DataFrame(grad_hess, columns=['xfl_grad_hess']), 
                              values], axis=1)
        elif grad is not None and hess is not None:
            data = pd.concat([pd.DataFrame(sample_index, columns=['xfl_id']),
                              pd.DataFrame(grad, columns=['xfl_grad']), 
                              pd.DataFrame(hess, columns=['xfl_hess']), values], axis=1)
        else:
            raise ValueError("Grad and hess are not given.")
        return Feature(data, list(values.columns))
            
    def slice_by_sample_index(self, sample_index: np.ndarray):
        df_sample_index = pd.DataFrame(sample_index)
        data = pd.merge(self.data, df_sample_index, left_on='xfl_id', right_on=0)
        return Feature(data, self.feature_columns)
    
    # def slice_by_row_index(self, row_index: np.ndarray):
    #     data = self.data.iloc[row_index]
    #     return Feature(data, self.feature_columns)
