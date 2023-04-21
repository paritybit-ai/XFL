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


from typing import Dict

import numpy as np

from common.communication.gRPC.python.channel import DualChannel, BroadcastChannel
from common.evaluation.metrics import ClusteringMetric
from common.utils.logger import logger
from service.fed_config import FedConfig
from service.fed_node import FedNode
from service.fed_control import _one_layer_progress, _update_progress_finish
from .api import get_table_agg_scheduler_inst
from .base import VerticalKmeansBase


class VerticalKmeansAssistTrainer(VerticalKmeansBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """

        Args:
                train_conf:
                *args:
                **kwargs:
        """
        self.channels = {}
        self.party_id_list = FedConfig.get_label_trainer() + FedConfig.get_trainer()

        sync_chann: Dict[str, DualChannel] = {}
        for party_id in self.party_id_list:
            sync_chann[party_id] = DualChannel(
                name="sync_" + party_id, ids=[FedNode.node_id, party_id])
        self.channels["sync"] = sync_chann
        conf = self._sync_config()
        train_conf.update(conf)

        super().__init__(train_conf, label=True, *args, **kwargs)

        self.DBI = None
        self.dist_sum = None

        cluster_res_chan: Dict[str, DualChannel] = {}
        tolerance_chan: Dict[str, DualChannel] = {}
        converged_flag_chan: Dict[str, DualChannel] = {}
        check_data_com: Dict[str, DualChannel] = {}

        for party_id in self.party_id_list:
            cluster_res_chan[party_id] = DualChannel(
                name="cluster_res_" + party_id, ids=[FedNode.node_id, party_id])
            tolerance_chan[party_id] = DualChannel(
                name="tolerance_" + party_id, ids=[FedNode.node_id, party_id])
            converged_flag_chan[party_id] = DualChannel(
                name="converged_flag_" + party_id, ids=[FedNode.node_id, party_id]
            )
            check_data_com[party_id] = DualChannel(
                name="check_data_com_" + party_id, ids=[FedNode.node_id, party_id]
            )

        self.channels["cluster_result"] = cluster_res_chan
        self.channels["tolerance"] = tolerance_chan
        self.channels["converged_flag"] = converged_flag_chan
        self.channels["check_data_com"] = check_data_com

        self.dist_table_agg_executor = get_table_agg_scheduler_inst(
            sec_conf=self.encryption, trainer_ids=FedConfig.get_label_trainer() +
            FedConfig.get_trainer()
        )

    def _sync_config(self):
        def count_key(conf):
            if isinstance(conf, dict):
                num = len(conf.keys())
                for k, v in conf.items():
                    num += count_key(v)
                return num
            else:
                return 0

        conf_to_update = {}
        max_key_num = 0
        for party_id in self.party_id_list:
            conf = self.channels["sync"][party_id].recv()
            num = count_key(conf)
            if num >= max_key_num:
                conf_to_update = conf
                max_key_num = num

        for party_id in self.party_id_list:
            self.channels["sync"][party_id].send(conf_to_update)

        return conf_to_update

    def init_centers(self):
        if self.init == "kmeans++":
            self.channels["init_prob"] = DualChannel(name="init_prob",
                                                     ids=[FedNode.node_id] + FedConfig.get_label_trainer())
            for i in range(1, self.k):
                dist_sum = self.dist_table_agg_executor.aggregate()
                p = (dist_sum ** 0.5).sum(axis=1)
                self.channels["init_prob"].send(p)

    @staticmethod
    def get_cluster(dist_sum):
        """
        Assign clustering results

        Args:
                dist_sum:

        Returns:

        """
        labels = np.argmin(dist_sum, axis=1)
        return np.array(labels)

    def check_data(self):
        lens, n_features = None, 0
        for party_id in self.party_id_list:
            recv = self.channels["check_data_com"][party_id].recv()
            m, n = recv
            if lens is None:
                lens = m
            elif lens != m:
                raise ValueError("Lengths of the train set mismatched.")
            n_features += n
        if n_features <= 0:
            raise ValueError("Number of the feature is zero. Stop training.")

    def fit(self):
        logger.info("vertical K-means scheduler training start.")
        self.check_data()
        self.init_centers()
        logger.info("{}::initialized centers.".format(self.identity))

        for iter_ in range(self.max_iter):

            if iter_ <= 0:
                self.dist_sum = self.dist_table_agg_executor.aggregate()

            cluster_result = self.get_cluster(self.dist_sum)
            for party_id in self.party_id_list:
                self.channels["cluster_result"][party_id].send(cluster_result)
            self.cluster_count_list = self.calc_cluster_count(cluster_result)

            tol_list = []
            for party_id in self.party_id_list:
                tol_list.append(self.channels["tolerance"][party_id].recv())
            tol_sum = sum(tol_list)

            logger.info("iter: {}, tol: {}.".format(iter_, tol_sum))
            self.is_converged = True if tol_sum < self.tol else False
            for party_id in self.party_id_list:
                self.channels["converged_flag"][party_id].send(
                    self.is_converged)

            self.dist_sum = self.dist_table_agg_executor.aggregate()

            self._calc_metrics(self.dist_sum, cluster_result, iter_)
            
            # calculate and update the progress of the training
            _one_layer_progress(iter_ + 1, self.max_iter)
            
            if self.is_converged:
                if iter_ + 1 < self.max_iter:
                    # update the progress of 100 to show the training is finished
                    _update_progress_finish()
                break

    def _calc_metrics(self, dist_sum, cluster_result, epoch):
        self.calc_dbi(dist_sum, cluster_result, epoch)

    def calc_dbi(self, dist_sum, cluster_result, epoch):
        dist_table = self.calc_ave_dist(dist_sum, cluster_result)
        if len(dist_table) == 1:
            raise ValueError(
                "DBI calculation error: All data are clustered into one group.")

        center_dist = self.dist_table_agg_executor.aggregate()

        cluster_avg_intra_dist = []
        for i in range(len(dist_table)):
            cluster_avg_intra_dist.append(dist_table[i][2])

        self.DBI = ClusteringMetric.calc_dbi(
            cluster_avg_intra_dist, center_dist)
        logger.info("epoch %d: dbi score=%.6g" % (epoch, self.DBI))

    def calc_ave_dist(self, dist_sum, cluster_result):
        distances_centers = dict()
        for vec, label in zip(dist_sum, cluster_result):
            if label not in distances_centers:
                distances_centers[label] = np.sqrt(vec[label])
            else:
                distances_centers[label] += np.sqrt(vec[label])
        calc_ave_dist_list = []
        for label in range(len(self.cluster_count_list)):
            count = self.cluster_count_list[label][1]
            if label not in distances_centers:
                calc_ave_dist_list.append([label, count, np.nan])
            else:
                calc_ave_dist_list.append(
                    [label, count, distances_centers[label] / count])
        return calc_ave_dist_list
