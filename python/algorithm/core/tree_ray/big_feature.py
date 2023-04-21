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
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @classmethod
    def create(cls,
               values: pd.DataFrame,
               indices: Optional[np.ndarray] = None,
               columns: Optional[list[str]] = None,
               grad: Optional[np.ndarray] = None,
               hess: Optional[np.ndarray] = None,
               grad_hess: Optional[np.ndarray] = None):
        # print(values.shape, len(indices) if indices is not None else None, "AAAAAAAAAA")
        # Note indices act on the Index column
        if indices is not None and len(indices) == 0:
            return None

        if indices is None and columns is None:
            selected_values = values
        elif indices is None:
            selected_values = values.loc[:, columns]
        elif columns is None:
            selected_values = values.loc[indices, :]
        else:
            # print("c")
            # print(len(indices), len(set(indices.tolist())))
            # print(indices, "HAHA")
            # print(values.index, "HBHBBH")
            # if len(indices) == values.shape[0]:
            #     print("ca")
            #     print(len(values.index.tolist()))
            #     selected_values = values.loc[:, columns]
            # else:
            #     print("cb")
            #     selected_values = values.loc[indices, columns]
            # selected_values = values.loc[:, columns]
            # print('cd')
            # print(indices, "----")
            selected_values = values.loc[indices, columns]
            # print('d')
            
        if indices is None:
            indices = values.index
        
        if grad_hess is not None:
            # data = pd.concat([pd.DataFrame(grad_hess, columns=['xfl_grad_hess']).set_index(indices),
            #                   selected_values], axis=1)
            data = pd.DataFrame(columns=['xfl_grad_hess'] + selected_values.columns.to_list(),
                                index=indices)
            data['xfl_grad_hess'] = grad_hess
            data[selected_values.columns] = selected_values
        else:
            # data = pd.concat([pd.DataFrame(grad, columns=['xfl_grad']).set_index(indices),
            #                   pd.DataFrame(hess, columns=['xfl_hess']).set_index(indices),
            #                   selected_values], axis=1)
            data = pd.DataFrame(columns=['xfl_grad', 'xfl_hess'] + selected_values.columns.to_list(),
                                index=indices)
            data['xfl_grad'] = grad
            data['xfl_hess'] = hess
            data[selected_values.columns] = selected_values
            
        return Feature(data)
            
    def slice_by_indices(self, indices: Optional[np.ndarray]):
        if indices is None:
            # Note it is not a copy.
            data = self.data
        else:
            data = self.data.loc[indices, :]
        return Feature(data)
    

if __name__ == "__main__":
    a = pd.DataFrame({
        "a": [1, 2, 3, 4], 
        "b": [3, 4, 5, 10],
        "c": [6, 11, 9, 10]
    }).set_index(np.array([2, 7, 11, 12]))
    
    grad = np.array([1, 2, 3])
    hess = np.array([3, 2, 1])
    
    indices = np.array([2, 11, 12])
    # indices = np.array([])
    columns = ['a', 'c']
    
    f = Feature.create(a, indices, columns, grad, hess, None)
    
    print(f.data)
    
    print(f.slice_by_indices(np.array([11, 12])).data)
    
    
    
    