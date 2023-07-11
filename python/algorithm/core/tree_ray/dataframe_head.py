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


from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import ray

from algorithm.core.tree_ray.dataset_info import RayDatasetInfo


class XgbDataFrameHead:
    def __init__(self,
                 dataset_info: RayDatasetInfo,
                 ray_tasks_num_returns: Optional[int] = None):
        self.dataset_info = dataset_info
        self.ray_tasks_num_returns = ray_tasks_num_returns
        self.ray_actors = list(dataset_info.actor_to_block_map.keys())
        self.NONE_REF = ray.put(None)
        self.LATEST_REF = ray.put('latest')
        self.ALL_REF = ray.put('all')
        
    def nunique(self, cols: Optional[List[Union[bool, int]]] = None):
        if cols is None:
            cols_ref = self.NONE_REF
        else:
            cols_ref = ray.put(cols)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.unique.remote(cols_ref))

        res = None
        
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            unique_df_list = ray.get(done_task)
            unique_df_list = list(filter(lambda x: False if x is None else True, unique_df_list))
            
            if len(unique_df_list) == 0:
                continue
            
            if res is None:
                res = pd.concat(unique_df_list)
            else:
                res = pd.concat([res] + unique_df_list)
            res = res.apply(lambda x: [np.unique(np.concatenate(x.tolist()))])
        
        res = res.apply(lambda x: len(x[0]))
        return res
    
    def set_cat_features(self, names: List[str]):
        names_ref = ray.put(names)
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.set_cat_features.remote(names_ref))
        # ray.get(ray_tasks)
        
    def set_split_points(self, split_points: Dict[str, list]):
        self.dataset_info.split_points = split_points
        split_points_ref = ray.put(split_points)
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.set_split_points.remote(split_points_ref))
        # ray.get(ray_tasks)
    
    def xgb_binning(self, num_bins: int):
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.xgb_binning_phase1.remote())
            
        def g(x: pd.Series):
            """ 1. x -- |(int, int), (int, int)|
                2. x -- |pd.Series, pd.Series|
            """
            x = x.tolist()

            if isinstance(x[0], pd.Series):
                # merge only supports two dataframe
                y = pd.merge(x[0], x[1], how='outer', left_index=True, right_index=True).fillna(0)
                counted_values = y[x[0].name+'_x'] + y[x[0].name+'_y']
                counted_values.name = x[0].name
                return [counted_values]
            else:
                a, b = x[0], x[1]
                min_v = min(a[0], b[0])
                max_v = max(a[1], b[1])
                return [(min_v, max_v)]

        # Get min, max for continuous features and value_count for categorial features
        statistic_df = None
        
        while len(ray_tasks):
            done_task, ray_tasks = ray.wait(ray_tasks)
            out = ray.get(done_task[0])
            
            if statistic_df is None:
                statistic_df = out
            else:
                statistic_df = pd.concat([statistic_df, out])
                statistic_df = statistic_df.apply(g)
        
        # min & max   -- |(min, max)|
        # value_count -- |pd.Series|
        
        # Calc split points or values in bins
        def f(x: pd.Series):
            x = x.iloc[0]

            if isinstance(x, pd.Series):  # Category
                if x.shape[0] > num_bins:
                    x.sort_values(ascending=False, inplace=True)
                    values = x.index.values.tolist()
                    values_unique = values[:num_bins - 1]
                    values_group = values[num_bins - 1:]
                    uniques = np.array(values_unique + [values_group], dtype=object)
                    return [uniques]
                else:
                    return [x.index.values.astype(object)]  # [x.index.values.astype(np.int64)]
            else:
                min_v, max_v = x
                split_points = np.linspace(min_v, max_v, num_bins + 1)[1:-1]
                return [split_points]
            
        split_points_df = statistic_df.apply(f)
        
        # Call actors to apply split points
        num_bins_ref = ray.put(num_bins)
        split_points_df_ref = ray.put(split_points_df)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.xgb_binning_phase2.remote(num_bins_ref, split_points_df_ref))
        ray.get(ray_tasks)
        
        if sum(split_points_df.shape) == 0:
            split_points = {}
        else:
            split_points = {
                feature_name: split_points_df[feature_name].iloc[0].tolist() for feature_name in split_points_df.columns
            }
        
        return split_points
        
