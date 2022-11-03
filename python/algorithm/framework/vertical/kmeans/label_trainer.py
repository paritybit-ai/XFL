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

from service.fed_config import FedConfig
from service.fed_node import FedNode
from common.communication.gRPC.python.channel import DualChannel
from .trainer import VerticalKmeansTrainer


class VerticalKmeansLabelTrainer(VerticalKmeansTrainer):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """
        Args:
            train_conf: training parameters
            *args:
            **kwargs:
        """
        super().__init__(train_conf, *args, **kwargs)

    def init_centers(self):
        init_center_chan: Dict[str, DualChannel] = {}
        for party_id in FedConfig.get_trainer():
            init_center_chan[party_id] = DualChannel(
                name="init_center_" + party_id, ids=[FedNode.node_id, party_id]
            )
        self.channels["init_center"] = init_center_chan

        if self.init == "random":
            center_ids = list(np.random.choice(len(self.train_features), self.k, replace=False))
            for party_id in FedConfig.get_trainer():
                self.channels["init_center"][party_id].send(center_ids)
            return center_ids
        elif self.init == "kmeans++":
            self.channels["init_prob"] = DualChannel(name="init_prob",
                                                     ids=[FedConfig.get_assist_trainer(), FedNode.node_id])
            center_ids = []
            while len(center_ids) < self.k:
                m = len(self.train_features)
                if len(center_ids) < 1:
                    center_ids.append(np.random.choice(m))
                else:
                    dist_table = self.distance_table(self.train_features.iloc[center_ids])
                    self.table_agg_executor.send(dist_table)
                    p = self.channels["init_prob"].recv()
                    p[center_ids] = 0
                    p = np.array(p / p.sum())
                    center_ids.append(np.random.choice(m, p=p))
                for party_id in FedConfig.get_trainer():
                    self.channels["init_center"][party_id].send(center_ids)
            return center_ids
