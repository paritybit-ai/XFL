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


import os
from pathlib import Path
from typing import List, Union, Optional, Dict

import pandas as pd
import numpy as np
import ray

from algorithm.core.tree_ray.assign_ray_input import gen_actor_indices
from algorithm.core.tree_ray.dataloader_head import XgbDataLoaderHead
from algorithm.core.tree_ray.dataframe_head import XgbDataFrameHead
from algorithm.core.tree_ray.tree_train_head import TreeTrainHead
from algorithm.core.tree_ray.dataset_info import RayDatasetInfo
from algorithm.core.tree_ray.xgb_actor import XgbActor
from algorithm.core.tree.tree_structure import BoostingTree, Tree, Node
from common.crypto.paillier.paillier import PaillierContext
from common.utils.logger import logger


class XgbRayHeadMaster:
    def __init__(self, ray_tasks_num_returns: Optional[int] = None):
        current_path = Path(os.path.dirname(__file__))
        package_path = current_path.parent.parent.parent
        ray.init(runtime_env={"working_dir": package_path, "excludes": ["core.*"]})
        num_cores = int(ray.cluster_resources()['CPU'])
        logger.info(f"Ray cluster core number: {num_cores}")
        
        self.ray_tasks_num_returns = ray_tasks_num_returns
        
        self.ray_actors = [
            XgbActor.options(num_cpus=1).remote() for i in range(num_cores)
        ]
        
        self.dataset_info: RayDatasetInfo = None
        self.val_dataset_info: RayDatasetInfo = None
        self.test_dataset_info: RayDatasetInfo = None
        
        self.ray_dataframe_head = None
        self.tree_train_head = TreeTrainHead(self.ray_actors, ray_tasks_num_returns)
        
    def scatter_data(self,
                     path_list: list[str],
                     dataset_type: str,
                     has_id: bool = True,
                     has_label: bool = True,
                     missing_values: Union[float, List[float]] = [],
                     atomic_row_size_per_cpu_core: int = 5000,
                     is_centralized: bool = True,
                     file_type: str = 'csv'):
        
        dataset_info = XgbDataLoaderHead.scatter_data(path_list,
                                                      dataset_type,
                                                      self.ray_actors,
                                                      has_id,
                                                      has_label,
                                                      missing_values,
                                                      atomic_row_size_per_cpu_core,
                                                      is_centralized,
                                                      file_type)
        if dataset_type == "train":
            self.dataset_info = dataset_info
        elif dataset_type == "val":
            self.val_dataset_info = dataset_info
        else:
            self.test_dataset_info = dataset_info
        return
        
    def nunique(self, cols: Optional[List[Union[bool, int]]] = None):
        if self.ray_dataframe_head is None:
            self.ray_dataframe_head = XgbDataFrameHead(self.dataset_info, self.ray_tasks_num_returns)
            
        res = self.ray_dataframe_head.nunique(cols)
        return res
    
    def set_cat_features(self, names: List[str]):
        if self.ray_dataframe_head is None:
            self.ray_dataframe_head = XgbDataFrameHead(self.dataset_info, self.ray_tasks_num_returns)
        
        self.ray_dataframe_head.set_cat_features(names)
        self.dataset_info.cat_names = names
        return
    
    def xgb_binning(self, num_bins: int):
        if self.ray_dataframe_head is None:
            self.ray_dataframe_head = XgbDataFrameHead(self.dataset_info, self.ray_tasks_num_returns)
        
        split_points = self.ray_dataframe_head.xgb_binning(num_bins)
        self.ray_dataframe_head.set_split_points(split_points)
        return split_points
    
    def sync_all_trees(self, boosting_tree: BoostingTree):
        self.tree_train_head.sync_all_trees(boosting_tree)
        return
        
    def sync_latest_tree(self, tree: Tree, lr: float, max_depth: int):
        self.tree_train_head.sync_latest_tree(tree, lr, max_depth)
        return
    
    def sync_config(self, 
                    paillier_context: PaillierContext,
                    cat_smooth: float,
                    lambda_: float):
        self.tree_train_head.sync_config(paillier_context,
                                         cat_smooth,
                                         lambda_)
        return
        
    def new_big_feature(self,
                        indices: Optional[np.ndarray],
                        columns: Optional[List[str]],
                        grad: Optional[np.ndarray],
                        hess: Optional[np.ndarray],
                        grad_hess: Optional[np.ndarray]):
        self.dataset_info.big_feature_names = columns
        indices_dict, grad_dict, hess_dict, grad_hess_dict = \
            self.tree_train_head.new_big_feature(indices,
                                                 columns,
                                                 grad,
                                                 hess,
                                                 grad_hess,
                                                 self.dataset_info)
        return indices_dict, grad_dict, hess_dict, grad_hess_dict
    
    def gen_big_feature_updater(self,
                                columns: Optional[List[str]]):
        self.dataset_info.big_feature_names = columns
        big_feature_updater = self.tree_train_head.gen_big_feature_updater(self.dataset_info.block_to_actor_map,
                                                                           columns)
        return big_feature_updater
        
    def gen_node_hist_iterator(self,
                               node_id: str,
                               packed: bool,
                               calc_count: bool,
                               indices: Optional[np.ndarray],
                               col_step: Optional[int] = None):
        """ Calc hist for a node with samples selected by indices

        Args:
            indices (Optional[np.ndarray]): indices of samples, if None, all the samples in big_feature are used.
            col_step (Optional[int], optional): num of features to calc in one step, if None, all features will be used
                in one step. Defaults to None.
        """
        num_features = len(self.dataset_info.big_feature_names)
        if col_step is not None and col_step >= num_features:
            col_step = None

        indices = gen_actor_indices(indices, self.dataset_info)
        
        hist_iterator = self.tree_train_head.gen_node_hist_iterator(node_id, packed, calc_count, indices, num_features, col_step)
        return hist_iterator

    def encrypt_grad_hess(self,
                          packed: bool,
                          context: PaillierContext,
                          precision: Optional[float]):
        iterator = self.tree_train_head.encrypt_grad_hess(packed=packed,
                                                          block_to_actor_map=self.dataset_info.block_to_actor_map,
                                                          context=context,
                                                          precision=precision,
                                                          lazy_return=True)
        return iterator
    
    def filter_sample_index(self,
                            node_id: str,
                            feature_name: str,
                            condition: Union[int, List[int]]):
        sample_index = self.tree_train_head.filter_sample_index(node_id, feature_name, condition)
        return sample_index
    
    def free_node_big_feature(self, node_id: str):
        self.tree_train_head.free_node_big_feature(node_id)
        return
    
    def calc_split_info(self,
                        is_remote: bool,
                        hist_dict: Dict[str, pd.DataFrame],
                        cat_names: List[str]):
        hint_split_info_iterator = self.tree_train_head.calc_split_info(is_remote,
                                                                        hist_dict,
                                                                        cat_names,
                                                                        lazy_return=True)
        return hint_split_info_iterator
    
    def make_indicator_for_prediction_on_tree(self, tree: Tree, local_party_id: str, dataset_type: str):
        indicator = self.tree_train_head.make_indicator_for_prediction_on_tree(tree, local_party_id, dataset_type)
        return indicator
    
    def make_indicator_for_prediction_on_boosting_tree(self, boosting_tree: BoostingTree, local_party_id: str, dataset_type: str):
        indicator = self.tree_train_head.make_indicator_for_prediction_on_boosting_tree(boosting_tree, local_party_id, dataset_type)
        return indicator
    
    def make_indicator_for_prediction_on_nodes(self, nodes: Dict[str, Node], dataset_type: str):
        if len(nodes) == 0:
            return {}
        indicator = self.tree_train_head.make_indicator_for_prediction_on_nodes(nodes, dataset_type)
        return indicator
    
    def predict_on_tree(self, tree: Tree, indicator: Dict[int, Dict[str, np.ndarray]], dataset_type: str):
        if dataset_type == "train":
            actor_to_block_map = self.dataset_info.actor_to_block_map
        elif dataset_type == "val":
            actor_to_block_map = self.val_dataset_info.actor_to_block_map
        elif dataset_type == "test":
            actor_to_block_map = self.test_dataset_info.actor_to_block_map
        prediction = self.tree_train_head.predict_on_tree(tree, indicator, actor_to_block_map) 
        return prediction
    
    def predict_on_boosting_tree(self, boosting_tree: BoostingTree, indicator: Dict[int, Dict[str, np.ndarray]], dataset_type: str):
        if dataset_type == "train":
            actor_to_block_map = self.dataset_info.actor_to_block_map
        elif dataset_type == "val":
            actor_to_block_map = self.val_dataset_info.actor_to_block_map
        elif dataset_type == "test":
            actor_to_block_map = self.test_dataset_info.actor_to_block_map
        prediction = self.tree_train_head.predict_on_boosting_tree(boosting_tree, indicator, actor_to_block_map)
        return prediction
    
        
if __name__ == "__main__":
    import time
    from pandas.testing import assert_frame_equal
    # path = ['/root/dataset/fate_dataset/fake_guest.csv',
    #         '/root/dataset/fate_dataset/fake_guest.csv']
    path = ['/root/dataset/fate_dataset/fake_guest.csv']
    # a = XgbRayHeadMaster(path, atomic_row_size_per_cpu_core=3300, is_centralized=True, file_type='csv')  # 3300)
    a = XgbRayHeadMaster()
    a.scatter_data(path,
                   dataset_type='train',
                   has_id=True,
                   has_label=True,
                   missing_values=[np.nan],
                   atomic_row_size_per_cpu_core=5000,
                   is_centralized=True,
                   file_type='csv')
    
    print(a.dataset_info.actor_to_block_map)

    res = a.nunique()
    print(res)

    res = a.nunique([2, 5, 10, 1, 8])
    print(res)
    print(res.to_numpy())
    print('ok')
    
    a.set_cat_features(['x2', 'x3', 'x9'])
    
    split_points = a.xgb_binning(num_bins=16)
    print(split_points)
    
    # a.set_loss_func('BCEWithLogitsLoss')
    
    print(a.dataset_info.block_to_actor_map)
    
    rows = sum([shape[0] for shape in a.dataset_info.shape])
    # indices = None
    # columns = None
    # # grad = 0.5 - a.dataset_info.label
    # grad = np.arange(len(a.dataset_info.label))
    # hess = grad * (1 - grad)
    
    # a.new_big_feature(indices,
    #                   columns,
    #                   grad,
    #                   hess,
    #                   None)
    
    indices = np.array([1, 2, 3, 4, 800, 900, 8000, 9000, 13000, 13001, 3300*6+1000, 3300*6+1100])
    columns = None
    # grad = 0.5 - a.dataset_info.label
    grad = np.arange(len(a.dataset_info.label))
    hess = grad * (1 - grad)
    grad = grad[indices]
    hess = hess[indices]
    
    a.new_big_feature(indices,
                      columns,
                      grad,
                      hess,
                      None)
    
    indices = np.array([1, 2, 3, 800, 900, 8000, 9000, 13000, 13001, 3300*6+1000, 3300*6+1100])
    batch_cols = None
    # a.calc_hist_for_node(indices, None)
    start = time.time()
    hist1 = a.calc_hist_for_node(indices, 10)
    print(time.time() - start)
    
    print(hist1)
    
    start = time.time()
    hist2 = a.calc_hist_for_node(indices, None)
    print(time.time() - start)
    
    # print(hist2)
    
    for k in hist1:
        assert_frame_equal(hist1[k], hist2[k])
    
#        xfl_grad   xfl_hess
#         sum        sum
# x2                    
# 3       1.0        0.0
# 5       3.0       -6.0
# 10      1.0        0.0
# 13      2.0       -2.0
# 15   1708.0 -1448306.0 -----
    

    
    # test_case
    # for statistic_df
    # import pandas as pd
    # df = pd.read_csv(path[0], index_col=0)
    
    # train_features = df.iloc[:, 1:].astype(np.float32)
    # # df[['x2', 'x3', 'x9']] = df[['x2', 'x3', 'x9']].astype('category')
    
    # out = train_features['x2'].value_counts().sort_index()
    # print(out)
    # print(out.shape)
    
    # assert (out == res['x2'][0]).all()
    # a, b = train_features['x1'].min(), train_features['x1'].max()
    # print(a, b)
    # assert a == res['x1'][0][0]
    # assert b == res['x1'][0][1]




# class A():
#     def __init__(self, a) -> None:
#         self.a = a
        
# class B(A):
#     def __init__(self, a) -> None:
#         super().__init__(a)
#         self.b = 1
        
# class C(A):
#     def __init__(self, a) -> None:
#         super().__init__(a)
#         self.c = 1
        
# class D(B, C):
#     def __init__(self, a) -> None:
#         super().__init__(a+1)
#         self.d = 1
        
# d = D(2)
# print(d.a)