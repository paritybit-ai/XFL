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


import random
from typing import Optional, Dict, List, Union

import ray
from ray.actor import ActorHandle
import numpy as np
import pandas as pd

from algorithm.core.tree_ray.assign_ray_input import gen_actor_input
from algorithm.core.tree_ray.xgb_actor import XgbTrainActor
from algorithm.core.tree_ray.dataset_info import RayDatasetInfo
from algorithm.core.tree.tree_structure import BoostingTree, Tree, Node
from common.crypto.paillier.paillier import PaillierContext


class TreeTrainHead:
    def __init__(self, ray_actors: list[XgbTrainActor], ray_tasks_num_returns: Optional[int] = None):
        self.ray_actors = ray_actors
        self.NONE_REF = ray.put(None)
        self.TRUE_REF = ray.put(True)
        self.FALSE_REF = ray.put(False)
        self.ray_tasks_num_returns = ray_tasks_num_returns
    
    def sync_all_trees(self, boosting_tree: BoostingTree):
        boosting_tree_ref = ray.put(boosting_tree)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.recv_boosting_tree.remote(boosting_tree_ref))
        # ray.get(ray_tasks)
        return
    
    def sync_latest_tree(self, tree: Tree, lr: float, max_depth: int):
        tree_ref = ray.put(tree)
        lr_ref = ray.put(lr)
        max_depth = ray.put(max_depth)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.recv_decision_tree.remote(tree_ref, lr_ref, max_depth))
        # ray.get(ray_tasks)
        return
    
    def sync_config(self, 
                    paillier_context: PaillierContext,
                    cat_smooth: float,
                    lambda_: float):
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.sync_config.remote(paillier_context,
                                                      cat_smooth,
                                                      lambda_))
        # ray.get(ray_tasks)
        return
        
    def new_big_feature(self,
                        indices: Optional[np.ndarray],
                        columns: Optional[List[str]],
                        grad: Optional[np.ndarray],
                        hess: Optional[np.ndarray],
                        grad_hess: Optional[np.ndarray],
                        dataset_info: RayDatasetInfo):
        indices, grad, hess, grad_hess = gen_actor_input(indices,
                                                         grad,
                                                         hess,
                                                         grad_hess,
                                                         dataset_info)
        
        indices_ref = {k: ray.put(v) if v is not None else self.NONE_REF for k, v in indices.items()}
        columns_ref = self.NONE_REF if columns is None else ray.put(columns)
        grad_ref = self.NONE_REF if grad is None else {k: ray.put(v) for k, v in grad.items()}
        hess_ref = self.NONE_REF if hess is None else {k: ray.put(v) for k, v in hess.items()}
        grad_hess_ref = self.NONE_REF if grad_hess is None else {k: ray.put(v) for k, v in grad_hess.items()}
        
        ray_tasks = []
        for actor in indices_ref:
            ray_tasks.append(actor.update_big_feature.remote(indices_ref[actor],
                                                             columns_ref,
                                                             grad_ref[actor] if isinstance(grad_ref, dict) else grad_ref,
                                                             hess_ref[actor] if isinstance(hess_ref, dict) else hess_ref,
                                                             grad_hess_ref[actor] if isinstance(grad_hess_ref, dict) else grad_hess_ref,
                                                             create_new=True))
        # ray.get(ray_tasks)
        
        out_indices = {}
        out_grad = {}
        out_hess = {}
        out_grad_hess = {}
        
        for actor in indices.keys():
            out_indices.update(indices[actor])
            if grad is not None:
                out_grad.update(grad[actor])
            if hess is not None:
                out_hess.update(hess[actor])
            if grad_hess is not None:
                out_grad_hess.update(grad_hess[actor])
        
        return out_indices, out_grad, out_hess, out_grad_hess
    
    def gen_big_feature_updater(self,
                                block_to_actor_map: Dict[int, ActorHandle],
                                columns: Optional[List[str]]):
        NONE_REF = self.NONE_REF
        
        class BigFeatureUpdater:
            def __init__(self, block_to_actor_map, columns):
                self.block_to_actor_map = block_to_actor_map
                self.columns_ref = NONE_REF if columns is None else ray.put(columns)
                self.is_created = {}
                
            def update(self,
                       indices: Dict[int, Optional[np.ndarray]],
                       grad: Optional[Dict[int, np.ndarray]],
                       hess: Optional[Dict[int, np.ndarray]],
                       grad_hess: Optional[Dict[int, np.ndarray]]):
                ray_tasks = []
                for block_id in indices.keys():
                    indices_ref = ray.put({block_id: indices[block_id]})
                    grad_ref = NONE_REF if grad is None else ray.put(grad)
                    hess_ref = NONE_REF if hess is None else ray.put(hess)
                    grad_hess_ref = NONE_REF if grad_hess is None else ray.put(grad_hess)
                    
                    actor = self.block_to_actor_map[block_id]

                    ray_tasks.append(actor.update_big_feature.remote(indices_ref,
                                                                     self.columns_ref,
                                                                     grad_ref,
                                                                     hess_ref,
                                                                     grad_hess_ref,
                                                                     create_new=not self.is_created.get(actor, False)))  # note it is judged by actor
                    self.is_created[actor] = True
                # ray.get(ray_tasks)
                return
            
        big_feature_updater = BigFeatureUpdater(block_to_actor_map, columns)
        return big_feature_updater
                    
    def _merge_hist(self, hist_dict_list: List[Dict[str, pd.DataFrame]]):
        hist_list_dict = {}
        for i, hist_dict in enumerate(hist_dict_list):
            if hist_dict is None:
                continue
            for col_name, hist in hist_dict.items():
                if col_name not in hist_list_dict:
                    hist_list_dict[col_name] = []
                hist_list_dict[col_name].append(hist_dict_list[i][col_name])

        num = min(len(self.ray_actors), len(hist_list_dict))
        ray_hist_dict_list = [{} for _ in range(num)]
        for i, feature_name in enumerate(hist_list_dict.keys()):
            ray_hist_dict_list[i % len(ray_hist_dict_list)].update({feature_name: hist_list_dict[feature_name]})
        ray_hist_dict_list = [ray.put(item) for item in ray_hist_dict_list]
        random.shuffle(ray_hist_dict_list)
        
        ray_tasks = []
        for i in range(num):
            actor = self.ray_actors[i]
            ray_tasks.append(actor.merge_hist.remote(ray_hist_dict_list[i]))
        
        if self.ray_tasks_num_returns is None:
            ray_tasks_num_returns = len(ray_tasks)
        else:
            ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
        
        out_hist_dict: Dict[str, pd.DataFrame] = {}
        while len(ray_tasks):
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            for out in out_list:
                out_hist_dict.update(out)
        return out_hist_dict
    
    def gen_node_hist_iterator(self,
                               node_id: str,
                               packed: bool,
                               calc_count: bool,
                               indices: Dict[ActorHandle, Dict[int, np.ndarray]],
                               num_features: Optional[int],
                               step: Optional[int]):
        node_id_ref = ray.put(node_id)
        packed_ref = self.TRUE_REF if packed else self.FALSE_REF
        calc_count_ref = self.TRUE_REF if calc_count else self.FALSE_REF
        indices_ref = {k: ray.put(v) if v is not None else self.NONE_REF for k, v in indices.items()}
        
        def gather_hist(node_id_ref, indices_ref, col_section_ref):
            ray_tasks = []
            for actor, indices_dict_ref in indices_ref.items():
                ray_tasks.append(actor.cal_hist_for_node.remote(node_id_ref,
                                                                packed_ref,
                                                                calc_count_ref,
                                                                indices_dict_ref,
                                                                col_section_ref))
                
            hist_dict_list = ray.get(ray_tasks)
            hist_dict: Dict[str, pd.DataFrame] = self._merge_hist(hist_dict_list)
            return hist_dict
            
        if step is None:
            hist = gather_hist(node_id_ref, indices_ref, self.NONE_REF)
            return [(hist, 0)]
        else:
            boundary_points = list(range(0, num_features, step)) + [num_features]
            
            class HistIterator:
                def __iter__(self):
                    self.boundaray_iterator = zip(boundary_points[:-1], boundary_points[1:])
                    self.round_left = len(boundary_points[:-1])
                    return self
                
                def __next__(self):
                    try:
                        a, b = next(self.boundaray_iterator)
                        col_section_ref = ray.put([a, b])
                        
                        hist = gather_hist(node_id_ref,
                                           indices_ref,
                                           col_section_ref)
                        self.round_left -= 1
                        return hist, self.round_left
                    except StopIteration:
                        raise StopIteration
                    
            return HistIterator()
    
    def encrypt_grad_hess(self,
                          packed: bool,
                          block_to_actor_map: Dict[int, ActorHandle],
                          context: PaillierContext,
                          precision: Optional[float],
                          lazy_return: bool):
        packed_ref = self.TRUE_REF if packed else self.FALSE_REF
        context_ref = ray.put(context)
        precision_ref = ray.put(precision)
        
        ray_tasks = []
        task_block_idx_map = {}

        for block_id in block_to_actor_map:
            block_id_ref = ray.put(block_id)
            ray_tasks.append(block_to_actor_map[block_id].encrypt_grad_hess.remote(packed_ref,
                                                                                   block_id_ref,
                                                                                   context_ref,
                                                                                   precision_ref))
            task_block_idx_map[ray_tasks[-1]] = block_id
            
        if lazy_return:
            class EncryptedGradHess:
                def __iter__(self):
                    self.ray_tasks = ray_tasks
                    self.task_block_idx_map = task_block_idx_map
                    return self
                
                def __next__(self):
                    if len(self.ray_tasks) != 0:
                        done_task, self.ray_tasks = ray.wait(self.ray_tasks)
                        block_idx = self.task_block_idx_map[done_task[0]]
                        out = ray.get(done_task[0])
                        return block_idx, out, len(self.ray_tasks)
                    else:
                        raise StopIteration
            return EncryptedGradHess()
        else:
            res = ray.get(ray_tasks)
            return [(None, res, 0)]
        
    def filter_sample_index(self,
                            node_id: str,
                            feature_name: str,
                            condition: Union[int, List[int]]):
        node_id_ref = ray.put(node_id)
        feature_name_ref = ray.put(feature_name)
        condition_ref = ray.put(condition)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.filter_sample_index.remote(node_id_ref,
                                                              feature_name_ref,
                                                              condition_ref))
        
        sample_index: Dict[str, list] = {}
        # import time
        # start = time.time()
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            for out in out_list:
                sample_index.update(out)
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # sample_index.update(out)
        # print(time.time() - start, '----')

        return sample_index
    
    def free_node_big_feature(self, node_id: str):
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.free_node_big_feature.remote(node_id))
        # ray.get(ray_tasks)
        return
    
    def calc_split_info(self,
                        is_remote: bool,
                        hist_dict: Dict[str, pd.DataFrame],
                        cat_names: List[str],
                        lazy_return: bool):
        is_remote_ref = self.TRUE_REF if is_remote else self.FALSE_REF
        cat_names_ref = ray.put(cat_names) if cat_names is not None else self.NONE_REF
        
        num = min(len(self.ray_actors), len(hist_dict))
        hist_dict_list = [{} for _ in range(num)]
        for i, feature_name in enumerate(hist_dict.keys()):
            hist_dict_list[i % len(hist_dict_list)].update({feature_name: hist_dict[feature_name]})
        hist_dict_ref_list = [ray.put(item) for item in hist_dict_list]
        
        random.shuffle(hist_dict_ref_list)
        
        ray_tasks = []
        for i in range(num):
            actor = self.ray_actors[i]
            ray_tasks.append(actor.calc_split_info.remote(is_remote_ref,
                                                          hist_dict_ref_list[i],
                                                          cat_names_ref))
            
        if lazy_return:
            ray_tasks_num_returns = self.ray_tasks_num_returns
            
            class HintSplitInfoIterator:
                def __iter__(self):
                    self.ray_tasks = ray_tasks
                    self.ray_tasks_num_returns = ray_tasks_num_returns
                    return self
                
                def __next__(self):
                    if len(self.ray_tasks) != 0:
                        if self.ray_tasks_num_returns is None:
                            ray_tasks_num_returns = len(self.ray_tasks)
                        else:
                            ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(self.ray_tasks))

                        done_task, self.ray_tasks = ray.wait(self.ray_tasks, num_returns=ray_tasks_num_returns)
                        hint_split_info_list = ray.get(done_task)
                        
                        best_hint_split_info = hint_split_info_list[0]
                        for hint_split_info in hint_split_info_list[1:]:
                            if hint_split_info['max_gain'] > best_hint_split_info['max_gain']:
                                best_hint_split_info = hint_split_info
                        # done_task, self.ray_tasks = ray.wait(self.ray_tasks)
                        # hint_split_info = ray.get(done_task[0])
                        # return hint_split_info, len(self.ray_tasks)
                        return best_hint_split_info, len(self.ray_tasks)
                    else:
                        raise StopIteration
            return HintSplitInfoIterator()
        else:
            hint_split_info = ray.get(ray_tasks)
            return [(hint_split_info, 0)]
        
    def make_indicator_for_prediction_on_tree(self, tree: Tree, local_party_id: str, dataset_type: str):
        tree_ref = ray.put(tree)
        local_party_id_ref = ray.put(local_party_id)
        dataset_type_ref = ray.put(dataset_type)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.make_indicator_for_prediction_on_tree.remote(tree_ref,
                                                                                local_party_id_ref,
                                                                                dataset_type_ref))
        
        indicator: Dict[int, Dict[str, np.ndarray]] = {}
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            
            for out in out_list:
                indicator.update(out)
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # indicator.update(out)
        return indicator
    
    def make_indicator_for_prediction_on_boosting_tree(self, boosting_tree: BoostingTree, local_party_id: str, dataset_type: str):
        boosting_tree_ref = ray.put(boosting_tree)
        local_party_id_ref = ray.put(local_party_id)
        dataset_type_ref = ray.put(dataset_type)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.make_indicator_for_prediction_on_boosting_tree.remote(boosting_tree_ref,
                                                                                         local_party_id_ref,
                                                                                         dataset_type_ref))
        
        indicator: Dict[int, Dict[str, np.ndarray]] = {}
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            
            for out in out_list:
                indicator.update(out)
                
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # indicator.update(out)
        return indicator
    
    def make_indicator_for_prediction_on_nodes(self, nodes: Dict[str, Node], dataset_type: str):
        nodes_ref = ray.put(nodes)
        dataset_type_ref = ray.put(dataset_type)
        
        ray_tasks = []
        for actor in self.ray_actors:
            ray_tasks.append(actor.make_indicator_for_prediction_on_nodes.remote(nodes_ref,
                                                                                 dataset_type_ref))
            
        indicator: Dict[int, Dict[str, np.ndarray]] = {}
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            
            for out in out_list:
                indicator.update(out)
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # indicator.update(out)
        return indicator
    
    def predict_on_tree(self,
                        tree: Tree,
                        indicator: Dict[int, Dict[str, np.ndarray]],
                        actor_to_block_map: Dict[ActorHandle, int]):
        tree_ref = ray.put(tree)
        
        ray_tasks = []
        for actor in self.ray_actors:
            indicator_ref = ray.put({block_id: indicator[block_id] for block_id in actor_to_block_map[actor]})
            ray_tasks.append(actor.predict_on_tree.remote(tree_ref, indicator_ref))
            
        prediction: Dict[int, np.ndarray] = {}
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            
            for out in out_list:
                prediction.update(out)
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # prediction.update(out)

        prediction = np.concatenate([prediction[i] for i in range(len(prediction))])
        return prediction
    
    def predict_on_boosting_tree(self,
                                 boosting_tree: BoostingTree,
                                 indicator: Dict[int, Dict[str, np.ndarray]],
                                 actor_to_block_map: Dict[ActorHandle, int]):
        boosting_tree_ref = ray.put(boosting_tree)
        
        ray_tasks = []
        for actor in self.ray_actors:
            indicator_ref = ray.put({block_id: indicator[block_id] for block_id in actor_to_block_map[actor]})
            ray_tasks.append(actor.predict_on_boosting_tree.remote(boosting_tree_ref, indicator_ref))
            
        prediction: Dict[int, np.ndarray] = {}
        while len(ray_tasks):
            if self.ray_tasks_num_returns is None:
                ray_tasks_num_returns = len(ray_tasks)
            else:
                ray_tasks_num_returns = min(self.ray_tasks_num_returns, len(ray_tasks))
            
            done_task, ray_tasks = ray.wait(ray_tasks, num_returns=ray_tasks_num_returns)
            out_list = ray.get(done_task)
            
            for out in out_list:
                prediction.update(out)
            # done_task, ray_tasks = ray.wait(ray_tasks)
            # out = ray.get(done_task[0])
            # prediction.update(out)

        prediction = np.concatenate([prediction[i] for i in range(len(prediction))])
        return prediction
