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

from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.tree.gain_calc import BestSplitInfo, cal_gain, cal_weight
from common.utils.logger import logger
from service.fed_control import ProgressCalculator


class HorizontalDecisionTreeAssistTrainer:
    def __init__(self,
                 tree_param: XGBTreeParam,
                 channels,
                 split_points,
                 cat_columns,
                 con_columns,
                 tree_index: Optional[int] = None
                 ):
        self.tree_param = tree_param
        self.channels = channels
        self.split_points = split_points
        self.tree_index = tree_index
        self.cat_columns = cat_columns
        self.con_columns = con_columns

    def _calc_global_hist(self):
        out = self.channels["agg_inst"].aggregate(average=False)
        G = out["G"]
        H = out["H"]
        
        # for feature_idx in self.cat_columns:
        #     cat_rank = cal_cat_rank(pd.Series(G[feature_idx]),
        #                             pd.Series(H[feature_idx]),
        #                             self.tree_param.cat_smooth)
        #     cat_rank.sort_values(inplace=True)
        #     G[feature_idx] = G[feature_idx][cat_rank.index.to_list()]
        #     H[feature_idx] = H[feature_idx][cat_rank.index.to_list()]
            
        self._find_best_split_point(G, H)

    def _find_best_split_point(self, G, H):
        best_split_info = BestSplitInfo()

        con_id_mapping = {v: k for k, v in enumerate(self.con_columns)}

        assert len(G) == len(H), "dimensions are inconsistent."
        for feature_idx in range(len(G)):
            idx = np.where((G[feature_idx] != 0) | (H[feature_idx] != 0))[0]
            
            cum_grad = np.cumsum(G[feature_idx][idx])
            cum_hess = np.cumsum(H[feature_idx][idx])
            gains = cal_gain(cum_grad, cum_hess, self.tree_param.lambda_)
            max_gain_index = np.argmax(gains)
            max_gain = gains[max_gain_index].item()

            if max_gain > best_split_info.gain:
                best_split_info.gain = max_gain
                best_split_info.feature_idx = feature_idx

                if feature_idx in self.cat_columns:
                    best_split_info.split_point = None
                    best_split_info.is_category = True
                    best_split_info.max_gain_index = int(idx[max_gain_index])
                else:
                    best_split_info.split_point = self.split_points[con_id_mapping.get(feature_idx)][max_gain_index]
                    best_split_info.max_gain_index = int(idx[max_gain_index])
                    best_split_info.is_category = False
                    best_split_info.left_cat = None

                left_weight = cal_weight(cum_grad[max_gain_index],
                                         cum_hess[max_gain_index],
                                         self.tree_param.lambda_).item()

                right_weight = cal_weight(cum_grad[-1] - cum_grad[max_gain_index],
                                          cum_hess[-1] - cum_hess[max_gain_index],
                                          self.tree_param.lambda_).item()

                best_split_info.left_bin_weight = left_weight
                best_split_info.right_bin_weight = right_weight

        if best_split_info.gain < self.tree_param.min_split_gain:
            for _, channel in self.channels["best_split_info_channel"].items():
                channel.send([1, best_split_info.gain])
            return

        for _, channel in self.channels["best_split_info_channel"].items():
            channel.send(best_split_info)

        sample_split_flag = True
        for _, channel in self.channels["sample_split_channel"].items():
            ss_flag = channel.recv()
            sample_split_flag = sample_split_flag and ss_flag

        for _, channel in self.channels["sample_split_channel"].items():
            channel.send(sample_split_flag)
        
        return

    def fit(self):
        logger.info("HorizontalDecisionTree::AssistTrainer start.")

        for depth in range(self.tree_param.max_depth):
            logger.info(f"HorizontalDecisionTree::AssistTrainer::Decision tree depth {depth} training start..")
            node_num = None
            for _, channel in self.channels["tree_structure_channel"].items():
                this_depth_node_num = channel.recv()
                if node_num is None:
                    node_num = this_depth_node_num
                elif node_num != this_depth_node_num:
                    raise ValueError(f"Node number in depth {depth} is inconsistent.")

            progress_calculator = ProgressCalculator(
                self.tree_param.num_trees,
                self.tree_param.max_depth,
                node_num
            )
            for node in range(node_num):
                self._calc_global_hist()
                progress_calculator.cal_custom_progress(
                    self.tree_index + 1, 
                    depth + 1, 
                    node + 1
                )
