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


import gc
import random
from typing import List, Union

import numpy as np
import pandas as pd
import ray

from algorithm.core.tree_ray.xgb_actor import XgbActor
from algorithm.core.tree_ray.dataset_info import RayDatasetInfo


def scatter_csv_data(path_list: list[str],
                     dataset_type: str,
                     ray_actors: list[XgbActor],
                     has_id: bool = True,
                     has_label: bool = True,
                     missing_values: Union[float, List[float]] = [],
                     atomic_row_size_per_cpu_core: int = 5000):
    shape = []
    col_names = []
    feature_names = []
    obj_ref_list = []
    label_name = None
    num_cols = 0
    global_rows = 0
    block_count = 0

    incomplete_df = None
    index_col = 0 if has_id else False
    
    dataset_info = RayDatasetInfo()
    dataset_info.actor_to_block_map = {id: [] for id in ray_actors}
    
    labels: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    
    for file_index, path in enumerate(path_list):
        iter_count = 0
        num_rows = 0

        if incomplete_df is None:
            df_iterators = [
                pd.read_csv(path,
                            index_col=index_col,
                            chunksize=atomic_row_size_per_cpu_core)
            ]
        else:
            skiprows = atomic_row_size_per_cpu_core - len(incomplete_df)

            df_iterators = [
                pd.read_csv(path,
                            index_col=index_col,
                            chunksize=skiprows),
            ]
            
            try:
                df_iterator2 = pd.read_csv(path,
                                           index_col=index_col,
                                           skiprows=range(1, skiprows+1),
                                           chunksize=atomic_row_size_per_cpu_core)
                df_iterators.append(df_iterator2)
            except pd.errors.EmptyDataError:
                pass

        df_index = 0
        
        file_type_ref = ray.put('csv')
        is_centralized_ref = ray.put(True)
        dataset_type_ref = ray.put(dataset_type)
        has_label_ref = ray.put(has_label)
        missing_values_ref = ray.put(missing_values)

        while True:
            try:
                if len(df_iterators) == 2 and iter_count == 1:
                    df_index = 1

                df: pd.DataFrame = next(df_iterators[df_index]).astype(np.float32)
                
                if df.shape[0] == 0:
                    break
                
                num_rows += len(df)
                global_rows += len(df)
                
                if has_label:
                    labels.append(df.iloc[:, 0].to_numpy())
                    
                indices.append(df.index.to_numpy())

                if iter_count == 0:
                    if file_index == 0:
                        num_cols = len(df.columns)
                        col_names = df.columns.tolist()
                        if has_label:
                            feature_names = col_names[1:]
                            label_name = col_names[0]
                        else:
                            feature_names = col_names
                            label_name = None
                    else:
                        if len(df.columns) != num_cols:
                            raise ValueError(
                                f"Number of columns of files provided are not the same. {num_cols} != {len(df.columns)}")

                        if df.columns.tolist() != col_names:
                            raise ValueError(
                                "Column names of files provided are not the same.")

                if incomplete_df is not None:
                    df = pd.concat([incomplete_df, df], axis=0)

                if len(df) == atomic_row_size_per_cpu_core:
                    block_index = block_count
                    sample_start_index = block_index * atomic_row_size_per_cpu_core
                    sample_indices = np.arange(sample_start_index, sample_start_index + atomic_row_size_per_cpu_core)
                    df.set_index(sample_indices, inplace=True)
                    obj_ref_list.append(ray.put([block_index, df]))
                    dataset_info.blocks_shape[block_index] = df.shape
                    incomplete_df = None
                    block_count += 1
                else:
                    incomplete_df = df

                if len(obj_ref_list) == len(ray_actors):
                    ray_tasks = []
                    for i, actor in enumerate(ray_actors):
                        ref = obj_ref_list[i]
                        ray_tasks.append(actor.recv_data.remote(ref,
                                                                file_type_ref,
                                                                is_centralized_ref,
                                                                dataset_type_ref,
                                                                has_label_ref,
                                                                missing_values_ref))
                        block_index = block_count - len(obj_ref_list) + i
                        dataset_info.actor_to_block_map[ray_actors[i]].append(block_index)

                    ray.get(ray_tasks)
                    obj_ref_list = []
                    gc.collect()
            except StopIteration:
                break

            iter_count += 1

        shape.append((num_rows, num_cols))

    if incomplete_df is not None:
        block_index = block_count
        sample_start_index = block_index * atomic_row_size_per_cpu_core
        sample_indices = np.arange(sample_start_index, sample_start_index + len(incomplete_df))
        incomplete_df.set_index(sample_indices, inplace=True)
        # sample_indices = np.arange(sample_start_index, sample_start_index + len(df))
        # df.set_index(sample_indices, inplace=True)
        obj_ref_list.append(ray.put([block_index, incomplete_df]))
        dataset_info.blocks_shape[block_index] = incomplete_df.shape
        block_count += 1

    if len(obj_ref_list) != 0:
        ray_tasks = []
        selected_actor_index = random.sample(
            range(len(ray_actors)), len(obj_ref_list))

        for i in range(len(obj_ref_list)):
            ref = obj_ref_list[i]
            actor_index = selected_actor_index[i]
            actor = ray_actors[actor_index]
            ray_tasks.append(actor.recv_data.remote(ref,
                                                    file_type_ref,
                                                    is_centralized_ref,
                                                    dataset_type_ref,
                                                    has_label_ref,
                                                    missing_values_ref))
            block_index = block_count - len(obj_ref_list) + i
            dataset_info.actor_to_block_map[ray_actors[actor_index]].append(block_index)

        ray.get(ray_tasks)
        
    dataset_info.feature_names = feature_names
    dataset_info.label_name = label_name
    dataset_info.shape = shape

    gc.collect()
    
    for actor_id, block_ids in dataset_info.actor_to_block_map.items():
        for id in block_ids:
            dataset_info.block_to_actor_map[id] = actor_id
            
    if has_label:
        dataset_info.label = np.concatenate(labels)
    dataset_info.indices = np.concatenate(indices)
    return dataset_info
