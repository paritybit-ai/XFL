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


from typing import List, Union

from algorithm.core.tree_ray.xgb_actor import XgbActor
from algorithm.core.tree_ray.csv_scatter import scatter_csv_data


class XgbDataLoaderHead:
    def __init__(self):
        super().__init__()
    
    @classmethod
    def scatter_data(cls,
                     path: Union[str, list[str]],
                     dataset_type: str,
                     ray_actors: list[XgbActor],
                     has_id: bool = True,
                     has_label: bool = True,
                     missing_values: Union[float, List[float]] = [],
                     atomic_row_size_per_cpu_core: int = 5000,
                     is_centralized: bool = True,
                     file_type: str = 'csv'):
        if is_centralized:
            if file_type == 'csv':
                dataset_info = \
                    scatter_csv_data(path,
                                     dataset_type,
                                     ray_actors,
                                     has_id,
                                     has_label,
                                     missing_values,
                                     atomic_row_size_per_cpu_core)
            else:
                raise NotImplementedError
        else:
            NotImplementedError
        return dataset_info
