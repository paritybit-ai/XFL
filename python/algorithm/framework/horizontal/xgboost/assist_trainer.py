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


import numpy as np

from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.encryption_param import get_encryption_param
# from algorithm.framework.vertical.kmeans.api import get_table_agg_scheduler_inst
from algorithm.core.horizontal.aggregation.api import get_aggregation_root_inst
from common.utils.logger import logger
from common.communication.gRPC.python.channel import DualChannel
from service.fed_config import FedConfig
from service.fed_control import ProgressCalculator
from .decision_tree_assist_trainer import HorizontalDecisionTreeAssistTrainer
from .common import Common


class HorizontalXgboostAssistTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        self.party_id_list = FedConfig.get_label_trainer() + FedConfig.get_trainer()
        self.split_points = None
        self.__init_channels()
        self.__init_cat_columns()

    def __init_cat_columns(self):
        for _, channel in self.cat_columns_channel.items():
            self.cat_columns = channel.recv()

    def __init_channels(self):
        # self.agg_inst = get_table_agg_scheduler_inst(
        #     sec_conf=self.encryption,
        #     trainer_ids=FedConfig.get_label_trainer() + FedConfig.get_trainer()
        # )
        self.agg_inst = get_aggregation_root_inst(
            sec_conf=self.encryption_params,
            root_id=FedConfig.get_assist_trainer(),
            leaf_ids=FedConfig.get_label_trainer() + FedConfig.get_trainer()
        )
        self.col_sample_channel = {}
        self.bin_split_points_channel = {}
        self.cat_columns_channel = {}
        self.best_split_info_channel = {}
        self.sample_split_channel = {}
        self.tree_structure_channel = {}
        self.restart_channel = {}
        self.early_stop_channel = {}
        for party_id in self.party_id_list:
            self.col_sample_channel[party_id] = DualChannel(
                name="col_sample_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.cat_columns_channel[party_id] = DualChannel(
                name="cat_columns_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.bin_split_points_channel[party_id] = DualChannel(
                name="bin_split_points_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.best_split_info_channel[party_id] = DualChannel(
                name="best_split_info_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.sample_split_channel[party_id] = DualChannel(
                name="sample_split_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.tree_structure_channel[party_id] = DualChannel(
                name="tree_structure_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.restart_channel[party_id] = DualChannel(
                name="restart_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
            self.early_stop_channel[party_id] = DualChannel(
                name="early_stop_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )
        self.channels = {
            "agg_inst": self.agg_inst,
            "best_split_info_channel": self.best_split_info_channel,
            "sample_split_channel": self.sample_split_channel,
            "tree_structure_channel": self.tree_structure_channel,
            "restart_channel": self.restart_channel,
            "early_stop_channel": self.early_stop_channel
        }

    def _continuous_feature_binning(self):
        min_vlist = []
        max_vlist = []
        
        for party_id in self.party_id_list:
            split_mat = self.bin_split_points_channel[party_id].recv()
            min_values = split_mat[0, :]
            max_values = split_mat[1, :]
            min_vlist.append(min_values)
            max_vlist.append(max_values)
    
        min_vlist = np.vstack(min_vlist)
        max_vlist = np.vstack(max_vlist)
        
        min_v = np.min(min_vlist, axis=0)
        max_v = np.max(max_vlist, axis=0)
        
        if True in (max_v == min_v):
            logger.debug("elements in max_v must be different with elements in min_v, but got max_v[i] == min_v[i]")
            raise ValueError("max_v == min_v")

        # (num_features, num_bins - 1)
        self.split_points = np.linspace(min_v, max_v, self.xgb_config.num_bins+1)[1:-1].T.tolist()

        for party_id in self.party_id_list:
            self.bin_split_points_channel[party_id].send(self.split_points)
            
    # def _continuous_feature_binning(self):
    #     table = self.agg_inst.aggregate()
    #     table /= len(self.party_id_list)
    #     table = torch.sort(table, axis=0).values
    #     self.split_points = table.T.tolist()
    #     for party_id in self.party_id_list:
    #         self.bin_split_points_channel[party_id].send(table)

    # def _cat_feature_binning(self):
    #     cat_max_values = np.zeros(len(self.cat_columns))
    #     for party_id in self.party_id_list:
    #         rec = self.cat_columns_channel[party_id].recv()
    #         cat_max_values = np.max([cat_max_values, rec], axis=0)
    #     for party_id in self.party_id_list:
    #         self.cat_columns_channel[party_id].send(cat_max_values)
    #     for _ in enumerate(self.cat_columns):
    #         table = self.agg_inst.aggregate()
    #         bins = [_[0] for _ in sorted(enumerate(table), key=lambda d: d[1], reverse=True)]
    #         bins = bins[:(self.xgb_config.num_bins - 1)]
    #         for party_id in self.party_id_list:
    #             self.cat_columns_channel[party_id].send(bins)

    def _feature_binning(self):
        self._continuous_feature_binning()
        # if len(self.cat_columns) > 0:
        #     self._cat_feature_binning()

    # def _calc_metric(self):
    #     cm = self.agg_inst.aggregate()
    #     print("global CM:", cm)

    def _col_sample(self):
        col_size = None
        for party_id in self.party_id_list:
            n = self.col_sample_channel[party_id].recv()
            if col_size is None:
                col_size = n
            elif col_size != n:
                raise ValueError("HorizontalXGBoost::AssistTrainer::features of trainers are inconsistent.")
        if 0 < self.xgb_config.subsample_feature_rate <= 1:
            sample_num = int(col_size * self.xgb_config.subsample_feature_rate)
        else:
            sample_num = col_size
        sampled_idx = np.sort(np.random.choice(col_size, sample_num, replace=False))
        feature_id_mapping = {a: b for a, b in enumerate(sampled_idx)}
        for party_id in self.party_id_list:
            self.col_sample_channel[party_id].send(feature_id_mapping)
        return feature_id_mapping

    def fit(self):
        logger.info("HorizontalXGBoost::AssistTrainer start.")
        self._feature_binning()

        for tree_idx in range(self.xgb_config.num_trees):
            logger.info("HorizontalXGBoost::AssistTrainer::Tree {} start training.".format(tree_idx))
            while True:
                restart_status = self.train_loop(tree_idx)
                # if all elements in restart_status are 0 or 2, then the training is finished
                if sum(restart_status) != len(restart_status):
                    break
                logger.info(f"trainer tree {tree_idx} training restart.")
            
            logger.info("Tree {} training done.".format(tree_idx))

            if 2 in restart_status:
                logger.info("trainer early stopped. because a tree's root is leaf.")
                break

            early_stop_flag = []
            for party_id in self.party_id_list:
                es = self.channels["early_stop_channel"][party_id].recv()
                # element in restart_status is 0, 1, or 2, and all elements should be the same
                early_stop_flag.append(es)
            if True in early_stop_flag:
                logger.info("assist trainer early stopped.")
                break
    
        # update the progress of 100 to show the training is finished
        ProgressCalculator.finish_progress()

    def train_loop(self, tree_idx):
        feature_id_mapping = self._col_sample()
        cat_columns_after_sampling = list(filter(
            lambda x: feature_id_mapping[x] in self.cat_columns, list(feature_id_mapping.keys())))
        con_columns_after_sampling = list(filter(
            lambda x: feature_id_mapping[x] not in self.cat_columns, list(feature_id_mapping.keys())))
        assist_trainer = HorizontalDecisionTreeAssistTrainer(
            tree_param=self.xgb_config,
            channels=self.channels,
            split_points=self.split_points,
            cat_columns=cat_columns_after_sampling,
            con_columns=con_columns_after_sampling,
            tree_index=tree_idx
        )
        tree = assist_trainer.fit()
        restart_status = []
        for party_id in self.party_id_list:
            rs = self.channels["restart_channel"][party_id].recv()
            # element in restart_status is 0, 1, or 2, and all elements should be the same
            restart_status.append(rs)
        
        return restart_status