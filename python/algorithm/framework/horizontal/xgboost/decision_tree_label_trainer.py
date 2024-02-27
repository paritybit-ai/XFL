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
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from algorithm.core.tree.feature_importance import FeatureImportance
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.tree.tree_structure import Node, SplitInfo, Tree
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.utils.logger import logger
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from service.fed_config import FedConfig
from service.fed_node import FedNode


class HorizontalDecisionTreeLabelTrainer:
    def __init__(self,
                 tree_param: XGBTreeParam,
                 y: np.ndarray,
                 y_pred: np.ndarray,
                 features: pd.DataFrame,
                 cat_columns: list,
                 cat_bins: list,
                 split_points: np.ndarray,
                 channels: Dict[str, Union[BroadcastChannel, DualChannel]],
                 tree_index: Optional[int] = None):
        self.tree_param = tree_param
        self.y = y
        self.y_pred = y_pred
        self.cat_columns = cat_columns
        self.cat_bins = cat_bins
        self.party_id = FedConfig.node_id
        self.features = features
        self.channels = channels
        self.split_points = split_points
        self.tree_index = tree_index
        loss_inst = get_xgb_loss_inst(list(self.tree_param.loss_param.keys())[0])
        self.grad = loss_inst.cal_grad(
            self.y, self.y_pred, after_prediction=True
        )
        self.hess = loss_inst.cal_hess(
            self.y, self.y_pred, after_prediction=True
        )
        self.feature_importance = {}
        self.feature_importance_type = tree_param.feature_importance_type

    def _calc_local_hist(self, node: Node):
        """
        calculate local histogram of gradients and hessians
        then send them for aggregation (sum)

        Args:
            node:

        Returns:

        """
        # feature values: range(xgb_config.num_bins)

        feature_slice = self.features.copy()
        feature_slice["grad"] = self.grad
        feature_slice["hess"] = self.hess

        if node.sample_index is None:
            # e.g., root node
            pass
        elif len(node.sample_index) == self.features.shape[0]:
            pass
        else:
            feature_slice = feature_slice.iloc[node.sample_index]

        G, H = [], []
        for f in feature_slice:
            if f == "grad" or f == "hess":
                continue
            grad_hist, hess_hist = [0] * self.tree_param.num_bins, [0] * self.tree_param.num_bins
            for k, v in feature_slice.groupby(f)["grad"].sum().to_dict().items():
                grad_hist[k] = v
            for k, v in feature_slice.groupby(f)["hess"].sum().to_dict().items():
                hess_hist[k] = v
            G.append(grad_hist)
            H.append(hess_hist)

        self.channels["agg_inst"].upload(
            {"G": np.array(G), "H": np.array(H)}, 1
        )

        return feature_slice

    def _calc_best_split(self, node: Node):
        """
        calc the best split point at current node

        Args:
            node: current node, i.e., begins with the root.

        Returns:

        """
        feature_slice = self._calc_local_hist(node)
        best_split_info = self.channels["best_split_info_channel"].recv()
        if isinstance(best_split_info, list):
            return best_split_info

        cat_id_mapping = {v: k for k, v in enumerate(self.cat_columns)}

        if best_split_info.is_category:
            left_cat = self.cat_bins[cat_id_mapping.get(best_split_info.feature_idx)][:best_split_info.max_gain_index + 1]
            best_split_info.left_cat = left_cat
            to_left = feature_slice.iloc[:, best_split_info.feature_idx].isin(left_cat)
        else:
            to_left = feature_slice.iloc[:, best_split_info.feature_idx] <= best_split_info.max_gain_index

        best_split_info.left_sample_index = feature_slice.index[to_left].to_list()
        best_split_info.right_sample_index = feature_slice.index[~to_left].to_list()

        best_split_info.num_left_bin = len(best_split_info.left_sample_index)
        best_split_info.num_right_bin = len(best_split_info.right_sample_index)

        sample_split_flag = True
        if min(best_split_info.num_left_bin, best_split_info.num_right_bin) < \
            self.tree_param.min_sample_split:
            sample_split_flag = False
        
        self.channels["sample_split_channel"].send(sample_split_flag)
        sample_split_flag = self.channels["sample_split_channel"].recv()
        if not sample_split_flag:
            return [2, best_split_info.num_left_bin, best_split_info.num_right_bin]
        
        return best_split_info

    def update_feature_importance(self, split_info: SplitInfo):
        inc_split, inc_gain = 1, split_info.gain

        # owner_id = split_info.owner_id
        fid = split_info.feature_idx
        owner_name = "All"

        if (owner_name, fid) not in self.feature_importance:
            self.feature_importance[(owner_name, fid)] = FeatureImportance(
                0, 0, self.feature_importance_type)

        self.feature_importance[(owner_name, fid)].add_split(inc_split)
        if inc_gain is not None:
            self.feature_importance[(owner_name, fid)].add_gain(inc_gain)

    def fit(self):
        # generate tree with a root node.
        tree = Tree(self.party_id, self.tree_index)
        logger.info(f"HorizontalDecisionTree::LabelTrainer::Decision tree {self.tree_index} training start..")

        for depth in range(self.tree_param.max_depth):
            logger.info(f"HorizontalDecisionTree::LabelTrainer::Decision tree depth {depth} training start..")
            # this_depth_nodes starts with [root_node]
            this_depth_nodes = tree.search_nodes(depth)

            self.channels["tree_structure_channel"].send(len(this_depth_nodes))
            for node in this_depth_nodes:
                logger.info(f"HorizontalDecisionTree::LabelTrainer::Depth {depth} - node {node.id} start training.")
                best_split_info = self._calc_best_split(node)
                if isinstance(best_split_info, list):
                    if best_split_info[0] == 1:
                        logger.info(f"HorizontalDecisionTree::LabelTrainer::Depth {depth} - node {node.id} stop training" + \
                                    f" since gain {best_split_info[1]:.6f} is less than min_split_gain {self.tree_param.min_split_gain}.")
                    elif best_split_info[0] == 2:
                        logger.info(f"HorizontalDecisionTree::LabelTrainer::Depth {depth} - node {node.id} stop training" + \
                                    f" since num_left_bin {best_split_info[1]} or num_right_bin {best_split_info[2]} is" + \
                                    f" less than min_sample_split {self.tree_param.min_sample_split} in at least one party.")
                    continue

                split_info = SplitInfo(owner_id=best_split_info.feature_owner,
                                       feature_idx=best_split_info.feature_idx,
                                       is_category=best_split_info.is_category,
                                       split_point=best_split_info.split_point,
                                       left_cat=best_split_info.left_cat,
                                       gain=best_split_info.gain)
                left_node_id, right_node_id = tree.split(node_id=node.id,
                                                         split_info=split_info,
                                                         left_sample_index=best_split_info.left_sample_index,
                                                         right_sample_index=best_split_info.right_sample_index,
                                                         left_sample_weight=best_split_info.left_bin_weight,
                                                         right_sample_weight=best_split_info.right_bin_weight)
                node.update_as_non_leaf(
                    split_info=split_info,
                    left_node_id=left_node_id,
                    right_node_id=right_node_id
                )

                self.update_feature_importance(split_info)

                logger.info(f"HorizontalDecisionTree::LabelTrainer::Depth {depth} - node {node.id} finish training.")
                gc.collect()

        logger.info(f"HorizontalDecisionTree::LabelTrainer::Decision tree {self.tree_index} training finished")
        return tree
    