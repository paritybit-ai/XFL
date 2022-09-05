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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pathos.pools import ThreadPool

from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.paillier_acceleration import embed, umbed
from algorithm.core.tree.big_feature import Feature
from algorithm.core.tree.feature_importance import FeatureImportance
from algorithm.core.tree.gain_calc import BestSplitInfo, cal_cat_rank, cal_gain, cal_weight
from algorithm.core.tree.goss import Goss
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.tree.tree_structure import Node, SplitInfo, Tree
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier, PaillierContext
from common.crypto.paillier.utils import get_core_num
from common.utils.constants import PAILLIER, PLAIN
from common.utils.logger import logger
from service.fed_config import FedConfig
from .debug_params import EMBEDING


class VerticalDecisionTreeLabelTrainer(object):
    def __init__(self,
                 tree_param: XGBTreeParam,
                 y: np.ndarray,
                 y_pred: np.ndarray,
                 features: pd.DataFrame,
                 cat_columns: list,
                 split_points: np.ndarray,
                 channels: Dict[str, Union[BroadcastChannel, DualChannel]],
                 encryption_context: Optional[PaillierContext] = None,
                 feature_id_mapping: Optional[Dict[int, int]] = None,
                 tree_index: Optional[int] = None):
        logger.info(
            f"Label trainer decision tree {tree_index} initialize start.")
        if tree_param.encryption_param.method not in [PAILLIER, PLAIN]:
            raise ValueError(
                f"Encryption method {tree_param.encryption_param.method} not supported.")

        self.tree_param = tree_param
        self.y = y
        self.y_pred = y_pred
        self.cat_columns = cat_columns
        self.split_points = split_points
        self.party_id = FedConfig.node_id
        self.max_num_cores = get_core_num(tree_param.max_num_cores)
        self.tree_index = tree_index

        loss_inst = get_xgb_loss_inst(self.tree_param.loss_param['method'])
        self.grad = loss_inst.cal_grad(
            self.y, self.y_pred, after_prediction=True)

        if tree_param.run_goss:
            goss = Goss(tree_param.top_rate, tree_param.other_rate)
            self.goss_selected_idx = goss.sampling(self.grad)
            hess = loss_inst.cal_hess(
                self.y[self.goss_selected_idx], self.y_pred[self.goss_selected_idx], after_prediction=True)
            self.hess = np.zeros_like(self.grad)
            self.hess[self.goss_selected_idx] = hess
            goss.update_gradients(self.grad, self.hess)
        else:
            self.hess = loss_inst.cal_hess(self.y, self.y_pred, after_prediction=True)
            self.goss_selected_idx = range(self.y.shape[0])

        sample_index = self.goss_selected_idx

        self.individual_grad_hess: BroadcastChannel = channels["individual_grad_hess"]
        self.tree_node_chann: BroadcastChannel = channels["tree_node"]

        self.summed_grad_hess_channs: Dict[str,
                                           DualChannel] = channels["summed_grad_hess"]
        self.min_split_info_channs: Dict[str,
                                         DualChannel] = channels["min_split_info"]
        self.sample_index_after_split_channs: Dict[str,
                                                   DualChannel] = channels["sample_index_after_split"]

        encryption_param = self.tree_param.encryption_param

        self.pri_context = encryption_context
        self.feature_importance = {}
        self.feature_importance_type = tree_param.feature_importance_type
        self.feature_id_mapping = feature_id_mapping

        if isinstance(encryption_param, PlainParam):
            self.individual_grad_hess.broadcast(
                [self.grad[sample_index], self.hess[sample_index]], use_pickle=True)
        elif isinstance(encryption_param, PaillierParam):
            num_cores = self.max_num_cores if encryption_param.parallelize_on else 1
            if EMBEDING:
                grad_hess = embed([self.grad[sample_index], self.hess[sample_index]], interval=(1 << 128), precision=64)
                enc_grad_hess = Paillier.encrypt(context=self.pri_context,
                                                 data=grad_hess,
                                                 precision=0,  # must be 0
                                                 obfuscation=True,
                                                 num_cores=num_cores)

                self.individual_grad_hess.broadcast(Paillier.serialize(enc_grad_hess, compression=False),
                                                    use_pickle=True)
            else:
                enc_grad = Paillier.encrypt(context=self.pri_context,
                                            data=self.grad[sample_index],
                                            precision=encryption_param.precision,
                                            obfuscation=True,
                                            num_cores=num_cores)

                enc_hess = Paillier.encrypt(context=self.pri_context,
                                            data=self.hess[sample_index],
                                            precision=encryption_param.precision,
                                            obfuscation=True,
                                            num_cores=num_cores)
                self.individual_grad_hess.broadcast(
                    [Paillier.serialize(enc_grad, compression=False), Paillier.serialize(enc_hess, compression=False)],
                    use_pickle=True)
        else:
            raise ValueError("Encryption param not supported.")

        if features.shape[1] == 0:
            self.big_feature = None
        else:
            self.big_feature = Feature.create(values=features.iloc[sample_index, :],
                                              sample_index=sample_index,
                                              grad=self.grad[sample_index],
                                              hess=self.hess[sample_index])
        logger.info(
            f"Label trainer decision tree {tree_index} initialize finished.")

    def _cal_local_best_split(self, node: Node):
        best_split_info = BestSplitInfo(feature_ower=self.party_id)

        if node.sample_index is None or len(node.sample_index) == self.big_feature.data.shape[0]:
            big_feature = self.big_feature
        else:
            big_feature = self.big_feature.slice_by_sample_index(node.sample_index)

        res_hist_list = []
        for col_name in big_feature.feature_columns:
            res_hist_list.append(big_feature.data.groupby([col_name])[['xfl_grad', 'xfl_hess']].agg({'sum'}))  # ({'count', 'sum'})
            
        # for categorial features, resort
        # cat column is count from the first col of cat feature
        for feature_idx in self.cat_columns:
            cat_rank = cal_cat_rank(res_hist_list[feature_idx][('xfl_grad', 'sum')],
                                    res_hist_list[feature_idx][('xfl_hess', 'sum')],
                                    self.tree_param.cat_smooth)
            cat_rank.sort_values(inplace=True)
            # index is saved in the Series's index
            res_hist_list[feature_idx] = res_hist_list[feature_idx].loc[cat_rank.index.to_list()]

        for feature_idx in range(len(res_hist_list)):
            res_hist_list[feature_idx] = res_hist_list[feature_idx].cumsum(axis=0)
            res_hist_list[feature_idx].rename(columns={"sum": "cum_sum"}, inplace=True)
            cum_grad = res_hist_list[feature_idx][('xfl_grad', 'cum_sum')].to_numpy()
            cum_hess = res_hist_list[feature_idx][('xfl_hess', 'cum_sum')].to_numpy()

            gains = cal_gain(cum_grad, cum_hess, self.tree_param.lambda_)
            
            if len(gains) == 1 and gains[0] == -np.inf:
                continue
            
            max_gain_index = np.argmax(gains)
            max_gain = gains[max_gain_index].item()

            if max_gain > best_split_info.gain:
                best_split_info.gain = max_gain
                best_split_info.feature_owner = self.party_id
                best_split_info.feature_idx = self.feature_id_mapping[feature_idx].item()
                
                # For categorial feature, split_point stores categories in left child branch
                if feature_idx in self.cat_columns:
                    # It is not much precise if some categorial values are not be sampled
                    left_cat = res_hist_list[feature_idx].index.to_list()[:max_gain_index + 1]
                    best_split_info.left_cat = []
                    for cat in left_cat:
                        ori_cat = self.split_points[feature_idx][cat]
                        if isinstance(ori_cat, list):
                            best_split_info.left_cat += ori_cat
                        else:
                            best_split_info.left_cat.append(ori_cat)
                    best_split_info.split_point = None
                    best_split_info.is_category = True
                    
                    filter = big_feature.data.iloc[:, feature_idx + 3].isin(left_cat)
                else:
                    # Because of sampling
                    max_split_index = int(res_hist_list[feature_idx][('xfl_grad', 'cum_sum')].index[max_gain_index])
                    max_split_index = min(max_split_index, len(self.split_points[feature_idx]) - 1)
                    
                    best_split_info.split_point = self.split_points[feature_idx][max_split_index]
                    best_split_info.left_cat = None
                    best_split_info.is_category = False
                    
                    filter = big_feature.data.iloc[:, feature_idx + 3] <= max_split_index
                    
                best_split_info.left_sample_index = big_feature.data[filter]['xfl_id'].tolist()
                best_split_info.right_sample_index = big_feature.data[~filter]['xfl_id'].tolist()

                left_weight = cal_weight(cum_grad[max_gain_index],
                                         cum_hess[max_gain_index],
                                         self.tree_param.lambda_).item()

                right_weight = cal_weight(cum_grad[-1] - cum_grad[max_gain_index],
                                          cum_hess[-1] - cum_hess[max_gain_index],
                                          self.tree_param.lambda_).item()

                best_split_info.left_bin_weight = left_weight
                best_split_info.right_bin_weight = right_weight
                best_split_info.num_left_bin = len(best_split_info.left_sample_index)
                best_split_info.num_right_bin = len(best_split_info.right_sample_index)
                best_split_info.max_gain_index = max_gain_index  # only valid for continuous feature
                    
        return best_split_info

    def _cal_remote_best_split(self) -> Dict[str, BestSplitInfo]:
        best_split_info_dict: Dict[str, BestSplitInfo] = {
            party_id: BestSplitInfo(feature_ower=party_id) for party_id in self.summed_grad_hess_channs
        }
        
        gain_infos: Dict[str, list] = {
            party_id: [] for party_id in self.summed_grad_hess_channs
        }
        
        is_continue_flags = np.array([True for party_id in self.summed_grad_hess_channs], dtype=np.bool)

        def decrypt_hist(hist_list: List[np.ndarray], num_cores: int, out_origin: bool = True) -> list:
            len_list = [len(item) for item in hist_list]
            cum_len = np.cumsum([0] + len_list)
            hist = np.concatenate(hist_list)
            hist = Paillier.decrypt(self.pri_context, hist, num_cores=num_cores, out_origin=out_origin)
            res = []
            for i in range(len(cum_len) - 1):
                res.append(hist[cum_len[i]: cum_len[i + 1]])
            return res

        while True:
            for i, party_id in enumerate(self.summed_grad_hess_channs):
                if not is_continue_flags[i]:
                    continue

                data = self.summed_grad_hess_channs[party_id].recv(use_pickle=True, wait=False)
                if data is None:
                    # Data has not been send, try it next round.
                    continue

                is_continue, grad_hess_hist_list, remote_cat_index = data

                if self.tree_param.encryption_param.method == PAILLIER:
                    if EMBEDING:
                        grad_hess_hist = []
                        count_hist_list = []
                        for item in grad_hess_hist_list:
                            grad_hess_hist.append(item[0])
                            count_hist_list.append(item[1])

                        grad_hess_hist = decrypt_hist(grad_hess_hist, num_cores=self.max_num_cores, out_origin=True)

                        grad_hist_list = []
                        hess_hist_list = []
                        for hist in grad_hess_hist:
                            a, b = umbed(hist, num=2, interval=(1 << 128), precison=64)
                            grad_hist_list.append(a)
                            hess_hist_list.append(b)
                    else:
                        grad_hist_list = []
                        hess_hist_list = []
                        count_hist_list = []
                        for item in grad_hess_hist_list:
                            grad_hist_list.append(item[0])
                            hess_hist_list.append(item[1])
                            count_hist_list.append(item[2])

                        grad_hist_list = decrypt_hist(grad_hist_list, num_cores=self.max_num_cores, out_origin=False)
                        hess_hist_list = decrypt_hist(hess_hist_list, num_cores=self.max_num_cores, out_origin=False)
                else:
                    grad_hist_list = []
                    hess_hist_list = []
                    count_hist_list = []
                    for item in grad_hess_hist_list:
                        grad_hist_list.append(item[0])
                        hess_hist_list.append(item[1])
                        count_hist_list.append(item[2])
                        
                for idx in range(len(grad_hess_hist_list)):
                    grad_hist, hess_hist, count_hist = \
                        np.array(grad_hist_list[idx], dtype=np.float32), np.array(hess_hist_list[idx], dtype=np.float32), np.array(count_hist_list[idx])
                    
                    # for categorial feature, resort
                    if idx in remote_cat_index:
                        cat_rank = cal_cat_rank(grad_hist, hess_hist, self.tree_param.cat_smooth)
                        cat_rank = np.argsort(cat_rank).tolist()
                        grad_hist = grad_hist[cat_rank]
                        hess_hist = hess_hist[cat_rank]
                        count_hist = count_hist[cat_rank]
                    else:
                        cat_rank = []
                
                    cum_grad_hist = np.cumsum(grad_hist)
                    cum_hess_hist = np.cumsum(hess_hist)

                    gains = cal_gain(cum_grad_hist, cum_hess_hist, self.tree_param.lambda_)
                    max_gain_index = np.argmax(gains)
                    max_gain = gains[max_gain_index].item()

                    num_left_sample = np.sum(count_hist[:max_gain_index + 1])
                    num_right_sample = np.sum(count_hist[max_gain_index + 1:])

                    info = {
                        'max_gain': max_gain,
                        'cum_grad': cum_grad_hist,
                        'cum_hess': cum_hess_hist,
                        'max_gain_index': max_gain_index,
                        "is_category": idx in remote_cat_index,
                        'cat_rank': cat_rank,
                        'num_left_sample': num_left_sample,
                        'num_right_sample': num_right_sample
                    }
                    gain_infos[party_id].append(info)
                    
                if not is_continue:
                    is_continue_flags[i] = is_continue
                    # No data will be send later, cal best_split_info
                    best_split_info: BestSplitInfo = best_split_info_dict[party_id]
                    for feature_idx, gain_info in enumerate(gain_infos[party_id]):
                        max_gain = gain_info["max_gain"]
                        cum_grad = gain_info["cum_grad"]
                        cum_hess = gain_info["cum_hess"]
                        max_gain_index = gain_info["max_gain_index"]
                        is_category = gain_info["is_category"]
                        cat_rank = gain_info["cat_rank"]

                        if max_gain > best_split_info.gain:
                            if len(cum_grad) == 1:
                                max_gain_split_index = 0
                            else:
                                max_gain_split_index = max_gain_index

                        if max_gain > best_split_info.gain:
                            best_split_info.gain = max_gain
                            best_split_info.feature_owner = party_id
                            best_split_info.feature_idx = feature_idx
                            best_split_info.split_point = None  # should not know
                            best_split_info.missing_value_on_left = None  # need not know
                            best_split_info.left_sample_index = None  # get it later
                            best_split_info.right_sample_index = None  # get it later

                            best_split_info.num_left_bin = gain_info["num_left_sample"]
                            best_split_info.num_right_bin = gain_info["num_right_sample"]

                            left_weight = cal_weight(cum_grad[max_gain_split_index],
                                                     cum_hess[max_gain_split_index],
                                                     self.tree_param.lambda_).item()

                            right_weight = cal_weight(cum_grad[-1] - cum_grad[max_gain_split_index],
                                                      cum_hess[-1] -
                                                      cum_hess[max_gain_split_index],
                                                      self.tree_param.lambda_).item()

                            best_split_info.left_bin_weight = left_weight
                            best_split_info.right_bin_weight = right_weight
                            best_split_info.max_gain_index = max_gain_index
                            # note this is not the final result of the left category
                            best_split_info.left_cat = [] if not cat_rank else cat_rank[:max_gain_index + 1]
                            best_split_info.is_category = is_category

            flag = np.any(is_continue_flags)
            if not flag:
                break
        gc.collect()
        return best_split_info_dict

    def get_feature_importance(self):
        return self.feature_importance

    def update_feature_importance(self, split_info: SplitInfo):
        inc_split, inc_gain = 1, split_info.gain

        owner_id = split_info.owner_id
        fid = split_info.feature_idx

        if (owner_id, fid) not in self.feature_importance:
            self.feature_importance[(owner_id, fid)] = FeatureImportance(
                0, 0, self.feature_importance_type)

        self.feature_importance[(owner_id, fid)].add_split(inc_split)
        if inc_gain is not None:
            self.feature_importance[(owner_id, fid)].add_gain(inc_gain)

    def fit(self) -> Tree:
        tree = Tree(self.party_id, self.tree_index)
        thread_pool = ThreadPool(2)
        logger.info(f"Decision tree {self.tree_index} training start..")
        if self.tree_param.run_goss:
            tree.root_node.sample_index = self.goss_selected_idx

        for depth in range(self.tree_param.max_depth):
            logger.info(f"Decision tree depth {depth} training start..")
            this_depth_nodes = tree.search_nodes(depth)

            for node in this_depth_nodes:
                logger.info(f"Depth {depth} - node {node.id} start training.")
                self.tree_node_chann.broadcast(node, use_pickle=True)
                best_split_info_dict: Dict[str, BestSplitInfo] = {}

                logger.info("Calculating local best split..")
                if self.big_feature is not None:
                    res1 = thread_pool.apipe(self._cal_local_best_split, node)
                logger.info("Calculating remote best split..")
                res2 = thread_pool.apipe(self._cal_remote_best_split)
                if self.big_feature is not None:
                    best_split_info_dict[self.party_id] = res1.get()
                logger.info("Calculating local best split done.")
                best_split_info_dict_remote = res2.get()
                logger.info("Calculating remote best split done.")
                best_split_info_dict.update(best_split_info_dict_remote)
                party_ids = list(best_split_info_dict.keys())

                best_split_party_id = party_ids[
                    np.argmax(
                        [best_split_info_dict[party_id].gain for party_id in party_ids])
                ]
                best_split_info = best_split_info_dict[best_split_party_id]

                if best_split_info.gain < self.tree_param.min_split_gain or \
                        min(best_split_info.num_left_bin,
                            best_split_info.num_right_bin) < self.tree_param.min_sample_split:
                    for party_id in self.min_split_info_channs:
                        self.min_split_info_channs[party_id].send([-1, -1, -1], use_pickle=True)
                    continue

                if best_split_info.feature_owner == self.party_id:
                    for party_id in self.min_split_info_channs:
                        self.min_split_info_channs[party_id].send([-1, -1, -1], use_pickle=True)
                    
                    split_info = SplitInfo(owner_id=best_split_info.feature_owner,
                                           feature_idx=best_split_info.feature_idx,
                                           is_category=best_split_info.is_category,
                                           split_point=best_split_info.split_point,
                                           left_cat=best_split_info.left_cat,
                                           gain=best_split_info.gain)
                else:
                    for party_id in self.min_split_info_channs:
                        if best_split_info.feature_owner == party_id:
                            self.min_split_info_channs[party_id].send(
                                [best_split_info.feature_idx, best_split_info.max_gain_index, best_split_info.left_cat],
                                use_pickle=True
                            )
                        else:
                            self.min_split_info_channs[party_id].send([-1, -1, -1], use_pickle=True)

                    split_info = SplitInfo(owner_id=best_split_info.feature_owner,
                                           feature_idx=best_split_info.feature_idx,
                                           gain=best_split_info.gain)

                    best_split_info.left_sample_index, best_split_info.right_sample_index = \
                        self.sample_index_after_split_channs[best_split_info.feature_owner].recv(
                            use_pickle=True)

                left_node_id, right_node_id = tree.split(node_id=node.id,
                                                         split_info=split_info,
                                                         left_sample_index=best_split_info.left_sample_index,
                                                         right_sample_index=best_split_info.right_sample_index,
                                                         left_sample_weight=best_split_info.left_bin_weight,
                                                         right_sample_weight=best_split_info.right_bin_weight)

                node.update_as_non_leaf(split_info=split_info,
                                        left_node_id=left_node_id,
                                        right_node_id=right_node_id)
                
                self.update_feature_importance(split_info)
                logger.info(f"Depth {depth} - node {node.id} finish training.")
                gc.collect()

        self.tree_node_chann.broadcast(None, use_pickle=True)
        logger.info(f"Decision tree {self.tree_index} training finished")
        return tree
