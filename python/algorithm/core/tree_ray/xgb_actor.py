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
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import ray

from algorithm.core.tree_ray.big_feature import Feature
from algorithm.core.tree.xgboost_loss import XGBLoss
from algorithm.core.tree.tree_structure import Tree, BoostingTree, Node
from algorithm.core.paillier_acceleration import embed, unpack
from algorithm.core.tree.gain_calc import cal_cat_rank, cal_gain, cal_weight
from common.crypto.paillier.paillier import Paillier, PaillierContext


"""
Note: Typings in Actors are for better understanding, they are ActorHandle actually.
"""


class XgbBaseActor:
    def __init__(self):
        self.features: Dict[int, pd.DataFrame] = {}
        self.label: Dict[int, Optional[np.ndarray]] = {}
        self.loss_method: Optional[str] = None
        self.loss_func: XGBLoss = None
        
        self.boosting_tree: Optional[BoostingTree] = None
        
        self.val_features: Dict[int, pd.DataFrame] = {}
        self.val_label: Dict[int, np.ndarray] = {}
        
        self.test_features: Dict[int, pd.DataFrame] = {}
        
        self.big_feature: Dict[int, Feature] = {}
        self.node_big_feature: Dict[str, Dict[int, Feature]] = {}
        
        self.split_points: Dict[str, list] = {}
        self.split_point_bin_map: Dict[str, Dict[float: int]] = {}
        self.paillier_context: PaillierContext = None
        self.cat_names: List[str] = []
        self.cat_smooth: float = None
        self.lambda_: float = None
        
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
    
    def report_actor_id(self):
        return self.actor_id

    def report_node_id(self):
        return self.node_id
    

class RayCentralCsvActor:
    def __init__(self):
        super().__init__()

    @classmethod
    def recv_data(cls,
                  data: list,  # [int, pd.DataFrame]
                  has_label: bool,
                  missing_values: List[float]):
        features: Dict[int, pd.DataFrame] = {}
        label: Dict[int, Optional[np.ndarray]] = {}
        
        if has_label:
            features[data[0]] = data[1].iloc[:, 1:]
            label[data[0]] = data[1].iloc[:, 0].to_numpy()
        else:
            features[data[0]] = data[1]
            label[data[0]] = None
            
        if missing_values != []:
            # features[data[0]].replace({k: 0 for k in missing_values}, inplace=True)
            features[data[0]] = features[data[0]].replace({k: 0 for k in missing_values})
            
        return features, label


class XgbDataFrameActor(XgbBaseActor):
    def __init__(self):
        super().__init__()
    
    def unique(self, cols: Optional[List[Union[bool, int]]] = None):
        res = None
        
        # def f(x: pd.Series):
        #     out = np.unique(np.concatenate([x.iloc[0], res[x.name].iloc[0]]))
        #     return [out]
        
        # although save memory, but much more slower
        # for _, df in self.features.items():
        #     ################################################################################################
        #     # Do not try to use lambda x: pd.unique(x)!
        #     # Return a numpy.ndarray is dangerous in apply body, when you are not sure whether
        #     # the returned array length are the same or not, it will result in different format
        #     # of dataframes.
        #     ################################################################################################
        #     if cols is not None:
        #         df = df.iloc[:, cols]
                
        #     unique_df = df.apply(lambda x: [np.unique(x.to_numpy())])
            
        #     if res is None:
        #         res = unique_df
        #     else:
        #         res = unique_df.apply(f)
        
        # Another version for cols=None, consumes more memory
        res = []
        for _, df in self.features.items():
            if cols is None:
                res.append(df)
            else:
                res.append(df[cols])
        if res == []:
            return None
        else:
            res = pd.concat(res).apply(lambda x: [np.unique(x.to_numpy())])
        return res
    
    def set_cat_features(self, names: List[str]):
        self.cat_names = names
        # for _, features in self.features.items():
        #     features[names] = features[names].astype('category')
        return
    
    def set_split_points(self, split_points: Dict[str, list]):
        self.split_points = split_points
        for feature_name in split_points:
            self.split_point_bin_map[feature_name] = {}
            for bin, split_point in enumerate(split_points[feature_name]):
                if isinstance(split_point, list):
                    for v in split_point:
                        self.split_point_bin_map[feature_name][v] = bin
                else:
                    self.split_point_bin_map[feature_name][split_point] = bin
        return
    
    def xgb_binning_phase1(self):
        def f(x: pd.Series):
            if x.name in self.cat_names:
                return [x.value_counts()]
            else:
                return [(x.min(), x.max())]
            # if x.dtypes == 'category':
            #     return [x.value_counts()]
            # else:
            #     return [(x.min(), x.max())]
            
        res = None
        
        def g(x: pd.Series):
            """ 1. x -- |(int, int)|
                2. x -- |pd.Series|
            """
            if isinstance(x.iloc[0], pd.Series):
                y = pd.merge(x.iloc[0], res[x.name].iloc[0], how='outer', left_index=True, right_index=True).fillna(0)
                counted_values = y[x.iloc[0].name+'_x'] + y[x.iloc[0].name+'_y']
                counted_values.rename(x.iloc[0].name, inplace=True)
                return [counted_values]
            else:
                a, b = x.iloc[0]
                c, d = res[x.name].iloc[0]
                min_v = min(a, c)
                max_v = max(b, d)
                return [(min_v, max_v)]
        
        for _, features in self.features.items():
            out = features.apply(f)
            
            if res is None:
                res = out
            else:
                res = out.apply(g)

        return res
    
    def xgb_binning_phase2(self,
                           num_bins: int,
                           split_points_df: pd.DataFrame):
        if num_bins <= 256:
            dtype = np.uint8
        elif num_bins <= 2 ** 16:
            dtype = np.uint16
        else:
            dtype = np.uint32
                
        def f(x: pd.Series):
            """1. Categorial_1 -- |object|
            #    2. Categorial_2 -- |np.int64|
               2. Continuous   -- |np.float64|
            """
            x = x.iloc[0]
            
            if x.dtype == object:
                if isinstance(x[-1], list):
                    value_map = {v: i for i, v in enumerate(x[:-1])}
                    value_map.update({v: len(x)-1 for v in x[-1]})
                else:
                    value_map = {v: i for i, v in enumerate(x)}
                return [value_map]
            else:
                bins = [-float('inf')] + x.tolist() + [float('inf')]
                return [bins]
            
        split_points_df = split_points_df.apply(f)
        
        def g(x: pd.Series):
            binning_info = split_points_df[x.name].iloc[0]
            if isinstance(binning_info, dict):
                codes = x.map(binning_info)
            else:
                codes = pd.cut(x, bins=binning_info, labels=range(len(binning_info)-1))
            return codes

        for block_idx, features in self.features.items():
            self.features[block_idx] = features.apply(g).astype(dtype)
        return


class XgbTrainActor(XgbBaseActor):
    def __init__(self):
        super().__init__()
        
    def recv_all_trees(self, boosting_tree: BoostingTree):
        self.boosting_tree = boosting_tree
    
    def recv_latest_tree(self, tree: Tree, lr: float, max_depth: int):
        self.boosting_tree.append(tree, lr, max_depth)
        
    def sync_config(self,
                    paillier_context: PaillierContext,
                    cat_smooth: float,
                    lambda_: float):
        self.paillier_context = paillier_context
        self.cat_smooth = cat_smooth
        self.lambda_ = lambda_
    
    def update_big_feature(self,
                           indices: Dict[int, Optional[np.ndarray]],
                           columns: Optional[List[str]],
                           grad: Optional[Dict[int, np.ndarray]],
                           hess: Optional[Dict[int, np.ndarray]],
                           grad_hess: Optional[Dict[int, np.ndarray]],
                           create_new: bool):
        # For label trainer, new a big_feature directly.
        # For trainer, the grad, hess or grad_hess are supposed to be ciphertext, so update a potion one time.
        if create_new:
            self.big_feature = {}
            gc.collect()
        
        for block_id, features in self.features.items():
            if indices is not None and block_id not in indices:
                # This block is not used because of sampling
                continue

            if grad_hess is None:
                self.big_feature[block_id] = Feature.create(values=features,
                                                            indices=None if indices is None else indices[block_id],
                                                            columns=columns,
                                                            grad=grad[block_id],
                                                            hess=hess[block_id],
                                                            grad_hess=None)
            else:
                self.big_feature[block_id] = Feature.create(values=features,
                                                            indices=None if indices is None else indices[block_id],
                                                            columns=columns,
                                                            grad=None,
                                                            hess=None,
                                                            grad_hess=grad_hess[block_id])
        return
    
    def cal_hist_for_node(self,
                          node_id: str,
                          packed: bool,
                          calc_count: bool,
                          indices: Optional[Dict[int, np.ndarray]],
                          col_section: Optional[Tuple[int, int]]):
        """ Calculate hist for this node_big_feature on selected feature columns.
            Note: Categorial feature hist is not sorted here.

        Args:
            node_id (str): node's id in a tree.
            packed (bool): if true, calc hist of column 'xfl_grad_hess', else, calc hist of columns 'xfl_grad' and 'xfl_hess'
            indices (Optional[Dict[int, np.ndarray]]): selected sample indices of the node.
                if indices is None, create a new self.node_big_feature equals to self.big_feature.
            col_section (Optional[Tuple[int, int]]): a section for feature columns on the node_big_feature.
            free_memory_after_execution: (bool): if true, delete the node_id key in self.node_big_feature and free the memroy.

        """
        if len(self.big_feature.keys()) == 0:
            return None
        
        if node_id not in self.node_big_feature:
            if indices is None:
                self.node_big_feature[node_id] = self.big_feature
            else:
                self.node_big_feature[node_id] = {}
                for block_idx in indices:
                    feature = self.big_feature[block_idx]
                    self.node_big_feature[node_id][block_idx] = feature.slice_by_indices(indices[block_idx])

        node_big_feature = self.node_big_feature[node_id]
        # hist_dict: Dict[int, Dict[str, pd.DataFrame]] = {}
        
        first_feature_col = 1 if packed else 2
        if col_section is None:
            columns = node_big_feature[list(node_big_feature.keys())[0]].data.columns.tolist()[first_feature_col:]  # for grad and hess
        else:
            columns = node_big_feature[
                list(node_big_feature.keys())[0]].data.columns.tolist()[col_section[0]+first_feature_col:col_section[1]+first_feature_col]

        agg_arg = {'sum', 'count'} if calc_count else {'sum'}
        
        # num_samples_sum = sum([feature.data.shape[0] for block_id, feature in node_big_feature.items()])
        
        # agg_feature = pd.DataFrame(columns=node_big_feature[list(node_big_feature.keys())[0]].data.columns,
        #                            index=range(num_samples_sum))
        
        agg_feature = pd.concat([feature.data for feature in node_big_feature.values()])
        
        hist: Dict[str: pd.DataFrame] = {name: None for name in columns}
        for name in columns:
            if not packed:
                hist[name] = agg_feature.groupby([name], observed=True)[['xfl_grad', 'xfl_hess']].agg(agg_arg)
            else:
                hist[name] = agg_feature.groupby([name], observed=True)[['xfl_grad_hess']].agg(agg_arg)
        
        # for block_idx, feature in node_big_feature.items():
        #     hist_dict[block_idx] = {}
        #     for name in columns:
        #         if not packed:
        #             res = feature.data.groupby([name], observed=True)[['xfl_grad', 'xfl_hess']].agg(agg_arg)
        #         else:
        #             res = feature.data.groupby([name], observed=True)[['xfl_grad_hess']].agg(agg_arg)
                    
        #         hist_dict[block_idx][name] = res
        
        # hist: Dict[str: pd.DataFrame] = {name: None for name in columns}
        
        # for col_name in hist:
        #     hist_list = [hist_dict[block_id][col_name] for block_id in hist_dict]
        #     if len(hist_list) == 1:
        #         hist[col_name] = hist_list[0]
        #     else:
        #         hist_df = pd.concat(hist_list)
        #         # numeric_only=False !!! Pandas bug.
        #         hist_df = hist_df.groupby(hist_df.index).sum(numeric_only=False)
        #         hist[col_name] = hist_df
        return hist
    
    def encrypt_grad_hess(self,
                          packed: bool,
                          block_id: int,
                          context: PaillierContext,
                          precision: Optional[float]):
        if block_id not in self.big_feature:
            # Actually not reach
            if packed:
                return np.array([])
            else:
                return [np.array([]), np.array([])]
        
        big_feature_df: pd.DataFrame = self.big_feature[block_id].data
        
        if packed:
            grad = big_feature_df["xfl_grad"].to_numpy()
            hess = big_feature_df["xfl_hess"].to_numpy()
            data = embed([grad, hess], interval=(1 << 128), precision=64)
            
            res = Paillier.encrypt(data=data,
                                   context=context,
                                   precision=0,  # must be 0 if data is packed grad and hess
                                   obfuscation=True,
                                   num_cores=1)
            
            res = Paillier.serialize(res, compression=False)
        else:
            data_grad = big_feature_df["xfl_grad"].to_numpy()
            data_hess = big_feature_df["xfl_hess"].to_numpy()
            data = [data_grad, data_hess]
            res = []
            for d in data:
                out = Paillier.encrypt(data=d,
                                       context=context,
                                       precision=precision,  # must be 0 if data is packed grad and hess
                                       obfuscation=True,
                                       num_cores=1)
                res.append(out)

            res = [Paillier.serialize(i, compression=False) for i in res]
        
        return res
    
    def filter_sample_index(self,
                            node_id: str,
                            feature_name: str,
                            condition: Union[int, List[int]]):
        """ 

        Args:
            node_id (str): node id
            feature_name (str): feature name
            condition (Union[int, List[int]]): if is cat feature, condition is List[int], else is int.
        """
        if node_id not in self.node_big_feature.keys():
            # No data in this actor
            return {}
        
        sample_index: Dict[int, list] = {}
        
        for block_id, feature in self.node_big_feature[node_id].items():
            # if feature.data[feature_name].dtype == 'category':
            if feature_name in self.cat_names:
                filter = feature.data[feature_name].isin(condition)
            else:
                filter = feature.data[feature_name] <= condition
                
            if len(feature.data[filter]) != 0:
                sample_index[block_id] = feature.data[filter].index.astype('int').tolist()

        return sample_index
    
    def free_node_big_feature(self, node_id: str):
        if node_id in self.node_big_feature:
            del self.node_big_feature[node_id]
            gc.collect()
        return
    
    def merge_hist(self, hist_list_dict: Dict[str, List[pd.DataFrame]]):
        out_hist_dict: Dict[str, pd.DataFrame] = {}
        
        for col_name in hist_list_dict:
            hist_df = pd.concat(hist_list_dict[col_name])
            # numeric_only=False !!! Pandas bug.
            hist_df = hist_df.groupby(hist_df.index).sum(numeric_only=False)
            out_hist_dict[col_name] = hist_df

        return out_hist_dict
    
    def calc_split_info(self,
                        is_remote: bool,
                        hist_dict: Dict[str, pd.DataFrame],
                        cat_names: Optional[List[str]]):
        def f(x):
            y = Paillier.decrypt(self.paillier_context, x, num_cores=1, out_origin=True)
            z = unpack(y, num=2)
            return z
        
        hint_split_info = {
            'max_gain': -float('inf'),
            "feature_name": None,  # fake name for remote party
            'split_bin': None,  # fake bin for remote party
            # "is_category": None,
            "left_cat": None,  # fake cat for remote party
            "left_weight": None,
            "right_weight": None,
            'num_left_sample': None,
            'num_right_sample': None
        }
        
        if not is_remote and cat_names is None:
            cat_names = self.cat_names
        
        for feature_name, feature_hist in hist_dict.items():
            if len(feature_hist) <= 1:
                # no split for this feature
                continue
            
            if is_remote and self.paillier_context:
                if ('xfl_grad_hess', 'sum') in feature_hist.columns:
                    feature_hist[('xfl_grad_hess', 'sum')] = feature_hist[('xfl_grad_hess', 'sum')].apply(f)
                    grad_hess_ndarray = np.array(feature_hist[('xfl_grad_hess', 'sum')].to_list()).astype(np.float32)
                    count = feature_hist[('xfl_grad_hess', 'count')].to_numpy()

                    feature_hist = pd.DataFrame(
                        np.concatenate([grad_hess_ndarray, count[:, np.newaxis]], axis=1),
                        index=feature_hist.index,
                        columns=pd.MultiIndex.from_tuples([('xfl_grad', 'sum'), ('xfl_hess', 'sum'), ('xfl_grad', 'count')])
                    )
                else:
                    feature_hist[[('xfl_grad', 'sum'), ('xfl_hess', 'sum')]] = \
                        feature_hist[[('xfl_grad', 'sum'), ('xfl_hess', 'sum')]].apply(lambda x: Paillier.decrypt(self.paillier_context, x, num_cores=1, out_origin=False))
                 
            is_category = feature_name in cat_names
            if is_category:
                cat_rank = cal_cat_rank(feature_hist[('xfl_grad', 'sum')],
                                        feature_hist[('xfl_hess', 'sum')],
                                        self.cat_smooth)
                cat_rank.sort_values(inplace=True)
                feature_hist = feature_hist.loc[cat_rank.index]
                
            feature_hist = feature_hist.cumsum(axis=0)
            feature_hist.rename(columns={"sum": "cum_sum", "count": "cum_count"}, inplace=True)
            cum_grad = feature_hist[('xfl_grad', 'cum_sum')].to_numpy()
            cum_hess = feature_hist[('xfl_hess', 'cum_sum')].to_numpy()

            gains = cal_gain(cum_grad, cum_hess, self.lambda_)
            max_gain_index = np.argmax(gains)
            feature_max_gain = gains[max_gain_index]
            
            if feature_max_gain > hint_split_info['max_gain']:
                count_hist = feature_hist[('xfl_grad', 'cum_count')]
                num_left_sample = count_hist.iloc[max_gain_index]
                num_right_sample = count_hist.iloc[-1] - count_hist.iloc[max_gain_index]
                
                if is_category:
                    self.out_left_cat = []
                    left_cat = feature_hist.index.to_list()[:max_gain_index + 1]
                    split_bin = None
                else:
                    left_cat = None
                    # convert to global index of split points of this feature, only for continuous feature
                    split_bin = int(feature_hist.index[max_gain_index])
                    
                left_weight = cal_weight(cum_grad[max_gain_index],
                                         cum_hess[max_gain_index],
                                         self.lambda_)

                right_weight = cal_weight(cum_grad[-1] - cum_grad[max_gain_index],
                                          cum_hess[-1] - cum_hess[max_gain_index],
                                          self.lambda_)
                
                hint_split_info['max_gain'] = feature_max_gain
                hint_split_info['feature_name'] = feature_name
                hint_split_info['split_bin'] = split_bin
                hint_split_info['left_cat'] = left_cat
                hint_split_info['is_category'] = is_category
                hint_split_info['left_weight'] = left_weight
                hint_split_info['right_weight'] = right_weight
                hint_split_info['num_left_sample'] = num_left_sample
                hint_split_info['num_right_sample'] = num_right_sample
                
        return hint_split_info
    
    def make_indicator_for_prediction_on_tree(self, tree: Tree, local_party_id: str, dataset_type: str):
        if dataset_type == "train":
            dataset = self.features
        elif dataset_type == "val":
            dataset = self.val_features
        elif dataset_type == 'test':
            dataset = self.test_features
        else:
            raise ValueError(f"Dataset type {dataset_type} is not valid, supported types are 'train' and 'val'.")
        
        # Dict[node_id, Dict[block_id, indicator]]
        indicator: Dict[str, Dict[int, np.ndarray]] = {}
        for node_id, node in tree.nodes.items():
            if not node.is_leaf and node.split_info.owner_id == local_party_id:
                indicator[node_id] = {}
                feature_name = node.split_info.feature_name
                
                if node.split_info.is_category:
                    if dataset_type == 'train':
                        left_cat = list(set([self.split_point_bin_map[feature_name][v] for v in node.split_info.left_cat]))
                    else:
                        left_cat = node.split_info.left_cat
                    for block_id, features in dataset.items():
                        data = features[feature_name].to_numpy()
                        indicator[node_id][block_id] = np.isin(data, left_cat)
                else:
                    if dataset_type == 'train':
                        split_point = self.split_point_bin_map[feature_name][node.split_info.split_point]
                        for block_id, features in dataset.items():
                            data = features[feature_name].to_numpy()
                            indicator[node_id][block_id] = (data <= split_point)
                    else:
                        split_point = node.split_info.split_point
                        for block_id, features in dataset.items():
                            data = features[feature_name].to_numpy()
                            indicator[node_id][block_id] = (data <= split_point)
        
        # Dict[node_id, Dict[block_id, indicator]] -> Dict[block_id, Dict[node_id, indicator]]
        out_indicator: Dict[int, Dict[str, np.ndarray]] = {}
        for node_id in indicator:
            for block_id, data in indicator[node_id].items():
                if block_id not in out_indicator:
                    out_indicator[block_id] = {}
                out_indicator[block_id][node_id] = data
        
        return out_indicator
        
    def make_indicator_for_prediction_on_boosting_tree(self, boosting_tree: BoostingTree, local_party_id: str, dataset_type: str):
        if dataset_type == "train":
            dataset = self.features
        elif dataset_type == "val":
            dataset = self.val_features
        elif dataset_type == 'test':
            dataset = self.test_features
        else:
            raise ValueError(f"Dataset type {dataset_type} is not valid, supported types are 'train' and 'val'.")
        
        # Dict[node_id, Dict[block_id, indicator]]
        indicator: Dict[str, Dict[int, np.ndarray]] = {}
        
        for tree in boosting_tree.trees:
            for node_id, node in tree.nodes.items():
                if not node.is_leaf and node.split_info.owner_id == local_party_id:
                    indicator[node_id] = {}
                    feature_name = node.split_info.feature_name
                    
                    if node.split_info.is_category:
                        if dataset_type == 'train':
                            left_cat = list(set([self.split_point_bin_map[feature_name][v] for v in node.split_info.left_cat]))
                        else:
                            left_cat = node.split_info.left_cat
                        for block_id, features in dataset.items():
                            data = features[feature_name].to_numpy()
                            indicator[node_id][block_id] = np.isin(data, left_cat)
                    else:
                        if dataset_type == 'train':
                            split_point = self.split_point_bin_map[feature_name][node.split_info.split_point]
                            for block_id, features in dataset.items():
                                data = features[feature_name].to_numpy()
                                indicator[node_id][block_id] = (data <= split_point)
                        else:
                            split_point = node.split_info.split_point
                            for block_id, features in dataset.items():
                                data = features[feature_name].to_numpy()
                                indicator[node_id][block_id] = (data <= split_point)
        
        # Dict[node_id, Dict[block_id, indicator]] -> Dict[block_id, Dict[node_id, indicator]]
        out_indicator: Dict[int, Dict[str, np.ndarray]] = {}
        for node_id in indicator:
            for block_id, data in indicator[node_id].items():
                if block_id not in out_indicator:
                    out_indicator[block_id] = {}
                out_indicator[block_id][node_id] = data
        
        return out_indicator
    
    def make_indicator_for_prediction_on_nodes(self, nodes: Dict[str, Node], dataset_type: str):
        if dataset_type == "train":
            dataset = self.features
        elif dataset_type == "val":
            dataset = self.val_features
        elif dataset_type == 'test':
            dataset = self.test_features
        else:
            raise ValueError(f"Dataset type {dataset_type} is not valid, supported types are 'train' and 'val'.")
        
        # Dict[node_id, Dict[block_id, indicator]]
        indicator: Dict[str, Dict[int, np.ndarray]] = {}
        for node_id, node in nodes.items():
            indicator[node_id] = {}
            feature_name = node.split_info.feature_name
            if node.split_info.is_category:
                if dataset_type == 'train':
                    left_cat = list(set([self.split_point_bin_map[feature_name][v] for v in node.split_info.left_cat]))
                else:
                    left_cat = node.split_info.left_cat
                for block_id, features in dataset.items():
                    data = features[feature_name].to_numpy()
                    indicator[node_id][block_id] = np.isin(data, left_cat)
            else:
                if dataset_type == 'train':
                    split_point = self.split_point_bin_map[feature_name][node.split_info.split_point]
                    for block_id, features in dataset.items():
                        data = features[feature_name].to_numpy()
                        indicator[node_id][block_id] = (data <= split_point)
                else:
                    split_point = node.split_info.split_point
                    for block_id, features in dataset.items():
                        data = features[feature_name].to_numpy()
                        indicator[node_id][block_id] = (data <= split_point)
                    
        # Dict[node_id, Dict[block_id, indicator]] -> Dict[block_id, Dict[node_id, indicator]]
        out_indicator: Dict[int, Dict[str, np.ndarray]] = {}
        for node_id in indicator:
            for block_id, data in indicator[node_id].items():
                if block_id not in out_indicator:
                    out_indicator[block_id] = {}
                out_indicator[block_id][node_id] = data
        return out_indicator
    
    def _gen_prediction(self, tree: Tree, indicator: Dict[str, np.ndarray]):
        num_samples = list(indicator.values())[0].shape[0]
        prediction = np.zeros((num_samples,), dtype=np.float32)
        depth = 0
        sample_in_node = {}

        while True:
            node_list = tree.search_nodes(depth)
            
            if not node_list:
                break

            for node in node_list:
                if node.is_leaf:
                    prediction[sample_in_node[node.id]] = node.weight
                else:
                    if depth == 0:
                        sample_in_node[node.left_node_id] = np.where(indicator[node.id] == 1)[0]
                        sample_in_node[node.right_node_id] = np.where(indicator[node.id] == 0)[0]
                    else:
                        sample_in_node[node.left_node_id] = np.intersect1d(
                            sample_in_node[node.id], np.where(indicator[node.id] == 1)[0])
                        sample_in_node[node.right_node_id] = np.intersect1d(
                            sample_in_node[node.id], np.where(indicator[node.id] == 0)[0])

            depth += 1
        return prediction
    
    def predict_on_tree(self, tree: Tree, indicator: Dict[int, Dict[str, np.ndarray]]):
        prediction: Dict[int, np.ndarray] = {}
        for block_id, indicator_dict in indicator.items():
            prediction[block_id] = self._gen_prediction(tree, indicator_dict)
        return prediction
    
    def predict_on_boosting_tree(self, boosting_tree: BoostingTree, indicator: Dict[int, Dict[str, np.ndarray]]):
        prediction: Dict[int, np.ndarray] = {}
        
        for tree_idx, tree in enumerate(boosting_tree.trees):
            for block_id, indicator_dict in indicator.items():
                p = self._gen_prediction(tree, indicator_dict)
                if block_id not in prediction:
                    prediction[block_id] = p * boosting_tree.lr[tree_idx]
                else:
                    prediction[block_id] += p * boosting_tree.lr[tree_idx]

        return prediction


@ray.remote(num_cpus=1)
class XgbActor(XgbDataFrameActor, XgbTrainActor):
    def __init__(self):
        super().__init__()
    
    def recv_data(self,
                  data,
                  file_type: str,
                  is_centralized: bool,
                  dataset_type: str,
                  has_label: bool,
                  missing_values: List[float]):
        if is_centralized:
            if file_type == 'csv':
                if dataset_type == 'train':
                    features, label = RayCentralCsvActor.recv_data(data,
                                                                   has_label,
                                                                   missing_values)
                    self.features.update(features)
                    self.label.update(label)
                elif dataset_type == 'val':
                    val_features, val_label = RayCentralCsvActor.recv_data(data,
                                                                           has_label,
                                                                           missing_values)
                    self.val_features.update(val_features)
                    self.val_label.update(val_label)
                else:
                    test_features, test_label = RayCentralCsvActor.recv_data(data,
                                                                             has_label,
                                                                             missing_values)
                    self.test_features.update(test_features)
                    # self.test_label.update(test_label)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return
        
        
        
        

