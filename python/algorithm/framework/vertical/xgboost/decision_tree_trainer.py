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
import math
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import ray

from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.big_feature import Feature
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.tree.tree_structure import Node, SplitInfo
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier, PaillierContext
from common.crypto.paillier.utils import get_core_num
from common.utils.constants import PAILLIER, PLAIN
from common.utils.logger import logger
from ray.internal.internal_api import free
from service.fed_config import FedConfig
from .debug_params import EMBEDING


class VerticalDecisionTreeTrainer(object):
    def __init__(self, 
                 tree_param: XGBTreeParam,
                 features: pd.DataFrame,
                 cat_columns: list,
                 split_points: np.ndarray,
                 channels: Dict[str, Union[BroadcastChannel, DualChannel]],
                 encryption_context: Optional[PaillierContext] = None,
                 feature_id_mapping: Optional[Dict[int, int]] = None,
                 tree_index: Optional[int] = None):
        logger.info(f"Trainer decision tree {tree_index} initialize start.")
        if tree_param.encryption_param.method not in [PAILLIER, PLAIN]:
            raise ValueError(f"Encryption method {tree_param.encryption_param.method} not supported.")
        
        self.tree_param = tree_param
        self.features = features
        self.cat_columns = cat_columns
        self.split_points = split_points
        self.party_id = FedConfig.node_id
        self.max_num_cores = get_core_num(tree_param.max_num_cores)
        self.tree_index = tree_index
        
        self.individual_grad_hess: BroadcastChannel = channels["individual_grad_hess"]
        self.tree_node_chann: BroadcastChannel = channels["tree_node"]
        
        self.summed_grad_hess_chann: DualChannel = channels["summed_grad_hess"]
        self.min_split_info_chann: DualChannel = channels["min_split_info"]
        self.sample_index_after_split_chann: DualChannel = channels["sample_index_after_split"]

        self.encryption_param = self.tree_param.encryption_param
        self.pub_context = encryption_context
        self.feature_id_mapping = feature_id_mapping

        if isinstance(self.encryption_param, PlainParam):
            self.grad, self.hess = self.individual_grad_hess.recv(use_pickle=True)
        elif isinstance(self.encryption_param, PaillierParam):
            if EMBEDING:
                self.grad_hess = self.individual_grad_hess.recv(use_pickle=True)
                self.grad_hess = Paillier.ciphertext_from(self.pub_context, self.grad_hess, compression=False)
            else:
                self.grad, self.hess = self.individual_grad_hess.recv(use_pickle=True)
                self.grad = Paillier.ciphertext_from(self.pub_context, self.grad, compression=False)  # ciphertext
                self.hess = Paillier.ciphertext_from(self.pub_context, self.hess, compression=False)
        else:
            raise ValueError("Encryption param not supported.")
        logger.info(f"Trainer decision tree {tree_index} initialize finished.")
        
    def fit(self) -> Dict[str, Node]:
        logger.info(f"Decision tree {self.tree_index} training start..")
        nodes = {}
        count = 0
        
        while True:
            node: Node = self.tree_node_chann.recv(use_pickle=True)
            
            if node is None:
                break
            
            logger.info(f"Node {node.id} training start..")
            if count == 0:
                if node.sample_index is None:
                    node.sample_index = range(self.features.shape[0])
                
                self.sample_index = node.sample_index
                if isinstance(self.encryption_param, PlainParam):
                    self.big_feature = Feature.create(values=self.features.iloc[node.sample_index, :],
                                                      sample_index=node.sample_index,
                                                      grad=self.grad,
                                                      hess=self.hess)
                else:
                    if EMBEDING:
                        self.big_feature = Feature.create(values=self.features.iloc[node.sample_index, :],
                                                          sample_index=node.sample_index,
                                                          grad_hess=self.grad_hess)
                    else:
                        self.big_feature = Feature.create(values=self.features.iloc[node.sample_index, :],
                                                          sample_index=node.sample_index,
                                                          grad=self.grad,
                                                          hess=self.hess)
                big_feature = self.big_feature

            else:
                big_feature = self.big_feature.slice_by_sample_index(node.sample_index)
                
            gc.collect()
                  
            count += 1
            
            logger.info(f"Node {node.id} finish preparing data.")
                
            def cal_grad_hess_hist_apart(col_name: str):
                res = big_feature.data.groupby([col_name])[['xfl_grad', 'xfl_hess']].agg({'count', 'sum'})
                return res
            
            logger.info(f"Node {node.id} calculating grad hess hist start..")
            send_times = math.ceil(len(big_feature.feature_columns) / self.tree_param.col_batch)
            res_hist_list = []
            if isinstance(self.encryption_param, PaillierParam):
                if EMBEDING:
                    for i in range(send_times):
                        # split by features
                        cat_index = list(
                            set(self.cat_columns).intersection(set(range(i * self.tree_param.col_batch, (i + 1) * self.tree_param.col_batch)))
                        )
                        
                        num = int(math.ceil(big_feature.data.shape[0] / self.tree_param.row_batch))
                        
                        for j in range(num):
                            data_id = ray.put(big_feature.data.iloc[self.tree_param.row_batch * j: self.tree_param.row_batch * (j + 1), :])
                            
                            @ray.remote
                            def cal_grad_hess_hist_embeding(col_name: str):
                                res = ray.get(data_id).groupby([col_name])['xfl_grad_hess'].agg({'count', 'sum'})
                                return res
                            
                            ray_tasks = []
                            for col_name in big_feature.feature_columns[i * self.tree_param.col_batch: (i + 1) * self.tree_param.col_batch]:
                                ray_tasks.append(cal_grad_hess_hist_embeding.remote(col_name))
                            b = ray.get(ray_tasks)
                            
                            free(data_id)
                            free(ray_tasks)

                            if j == 0:
                                res = b
                            else:
                                b_id = ray.put(b)
                                res_id = ray.put(res)
                                
                                @ray.remote
                                def merge_embeding(k):
                                    res = ray.get(res_id)
                                    b = ray.get(b_id)
                                    r = pd.merge(res[k], b[k], how='outer', left_index=True, right_index=True).fillna(0)
                                    r = pd.Series(b[k].columns).apply(lambda x: r[x+'_x'] + r[x+'_y']).T
                                    r.columns = list(b[k].columns)
                                    return r
                                
                                ray_tasks = [merge_embeding.remote(k) for k in range(len(b))]
                                res = ray.get(ray_tasks)
                                free(b_id)
                                free(res_id)
                                free(ray_tasks)
                            gc.collect()
                            
                        res_hist_partial_list = res
                        hist_list = [(res_hist['sum'].to_numpy(), res_hist['count'].to_numpy()) for res_hist in res_hist_partial_list]

                        if (i + 1) == send_times:
                            # 【stop_flag, hist_list, index of category feature(in binary form)]
                            self.summed_grad_hess_chann.send([False, hist_list, cat_index], use_pickle=True)
                        else:
                            self.summed_grad_hess_chann.send([True, hist_list, cat_index], use_pickle=True)
                        
                        res_hist_list += res_hist_partial_list
                        gc.collect()
                else:
                    for i in range(send_times):
                        cat_index = list(
                            set(self.cat_columns).intersection(set(range(i * self.tree_param.col_batch, (i + 1) * self.tree_param.col_batch)))
                        )
                        
                        num = int(math.ceil(big_feature.data.shape[0] / self.tree_param.row_batch))
                        for j in range(num):
                            data_id = ray.put(big_feature.data.iloc[self.tree_param.row_batch * j: self.tree_param.row_batch * (j + 1), :])
                            
                            @ray.remote
                            def cal_grad_hess_hist(col_name: str):
                                res = ray.get(data_id).groupby([col_name])[['xfl_grad', 'xfl_hess']].agg({'count', 'sum'})
                                return res
                            
                            ray_tasks = []
                            for col_name in big_feature.feature_columns[i * self.tree_param.col_batch: (i + 1) * self.tree_param.col_batch]:
                                ray_tasks.append(cal_grad_hess_hist.remote(col_name))
                            b = ray.get(ray_tasks)
                            
                            free(data_id)
                            free(ray_tasks)

                            if j == 0:
                                res = b
                            else:
                                b_id = ray.put(b)
                                res_id = ray.put(res)
                                
                                @ray.remote
                                def merge(k):
                                    res = ray.get(res_id)
                                    b = ray.get(b_id)
                                    r = pd.merge(res[k], b[k], how='outer', left_index=True, right_index=True).fillna(0)
                                    r = pd.Series(list(b[k].columns)).apply(lambda x: r[(x[0]+'_x', x[1])] + r[(x[0]+'_y', x[1])]).T
                                    r.columns = pd.MultiIndex.from_tuples(b[k].columns)
                                    return r
                                
                                ray_tasks = [merge.remote(k) for k in range(len(b))]
                                res = ray.get(ray_tasks)
                                free(b_id)
                                free(res_id)
                                free(ray_tasks)
                            gc.collect()
                            
                        res_hist_partial_list = res
                        hist_list = [(res_hist[('xfl_grad', 'sum')].to_numpy(),
                                      res_hist[('xfl_hess', 'sum')].to_numpy(), 
                                      res_hist[('xfl_grad', 'count')].to_numpy()) for res_hist in res_hist_partial_list]
                        if (i + 1) == send_times:
                            self.summed_grad_hess_chann.send([False, hist_list, cat_index], use_pickle=True)
                        else:
                            self.summed_grad_hess_chann.send([True, hist_list, cat_index], use_pickle=True)
                        
                        res_hist_list += res_hist_partial_list
                        gc.collect()
            else:
                cat_index = self.cat_columns
                res_hist_list = pd.Series(big_feature.feature_columns).apply(cal_grad_hess_hist_apart)
                hist_list = [(res_hist[('xfl_grad', 'sum')].to_numpy(),
                              res_hist[('xfl_hess', 'sum')].to_numpy(),
                              res_hist[('xfl_grad', 'count')].to_numpy()) for res_hist in res_hist_list]
                self.summed_grad_hess_chann.send([False, hist_list, cat_index], use_pickle=True)
                gc.collect()
                
            logger.info(f"Node {node.id} calculating grad hess hist finished")

            feature_idx, max_gain_index, left_cat = self.min_split_info_chann.recv(use_pickle=True)
            if feature_idx == -1:
                continue
            
            if feature_idx in self.cat_columns:
                if isinstance(self.encryption_param, PaillierParam) and EMBEDING:
                    left_cat_index = res_hist_list[feature_idx]['sum'].index[left_cat]
                else:
                    left_cat_index = res_hist_list[feature_idx][('xfl_grad', 'sum')].index[left_cat]
                
                left_cat_values_ori = [self.split_points[feature_idx][index] for index in left_cat_index]
                
                left_cat_values = []
                for cat in left_cat_values_ori:
                    if isinstance(cat, list):
                        left_cat_values += cat
                    else:
                        left_cat_values.append(cat)
            
                split_info = SplitInfo(owner_id=self.party_id,
                                       feature_idx=self.feature_id_mapping[feature_idx].item(),
                                       is_category=True,
                                       split_point=None,
                                       left_cat=left_cat_values)
                
                if isinstance(self.encryption_param, PaillierParam) and EMBEDING:
                    left_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 2].isin(left_cat)]['xfl_id'].tolist()
                    right_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 2].isin(left_cat)]['xfl_id'].tolist()
                else:
                    left_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 3].isin(left_cat)]['xfl_id'].tolist()
                    right_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 3].isin(left_cat)]['xfl_id'].tolist()
            else:
                if isinstance(self.encryption_param, PaillierParam) and EMBEDING:
                    # works when goss is true
                    split_point_index = int(res_hist_list[feature_idx]['sum'].index[max_gain_index])
                else:
                    split_point_index = int(res_hist_list[feature_idx][('xfl_grad', 'sum')].index[max_gain_index])

                # may not be necessary, just for safe
                split_point_index = min(split_point_index, len(self.split_points[feature_idx]) - 1)
                split_point = self.split_points[feature_idx][split_point_index]

                split_info = SplitInfo(owner_id=self.party_id,
                                       feature_idx=self.feature_id_mapping[feature_idx].item(),
                                       is_category=False,
                                       split_point=split_point,
                                       left_cat=None)
            
                if isinstance(self.encryption_param, PaillierParam) and EMBEDING:
                    left_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 2] <= split_point_index]['xfl_id'].tolist()
                    right_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 2] > split_point_index]['xfl_id'].tolist()
                else:
                    left_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 3] <= split_point_index]['xfl_id'].tolist()
                    right_sample_index = big_feature.data[big_feature.data.iloc[:, feature_idx + 3] > split_point_index]['xfl_id'].tolist()
            
            nodes[node.id] = Node(id=node.id, split_info=split_info, is_leaf=False)
            
            self.sample_index_after_split_chann.send([left_sample_index, right_sample_index], use_pickle=True)
            logger.info(f"Node {node.id} training finished.")
        return nodes
        