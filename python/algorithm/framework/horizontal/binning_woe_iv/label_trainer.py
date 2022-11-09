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

import numpy as np
import pandas as pd

from algorithm.core.data_io import CsvReader
from algorithm.core.horizontal.aggregation.api import get_aggregation_leaf_inst
from common.communication.gRPC.python.channel import DualChannel
from common.crypto.key_agreement.diffie_hellman import DiffieHellman
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from typing import OrderedDict

from service.fed_config import FedConfig


class HorizontalBinningWoeIvLabelTrainer(object):
    def __init__(self, train_conf: dict):
        self.config = TrainConfigParser(train_conf)
        self.aggregation = self.config.train_params.get("aggregation", {})
        self.encryption = self.aggregation.get("encryption")
        self.data = self.init_data()
        self.y = pd.DataFrame(self.data.label(), columns=["y"])
        self.leaf_ids = FedConfig.get_label_trainer() + FedConfig.get_trainer()
        # generate random number
        if self.encryption["method"] == "otp" and len(self.leaf_ids) == 2:
            # key exchange
            key_exchange_conf = self.encryption["key_exchange"]
            self.random_protocol = DiffieHellman(self.leaf_ids,
                                                 key_bitlength=key_exchange_conf['key_bitlength'],
                                                 optimized=key_exchange_conf["optimized"],
                                                 channel_name="diffie_hellman_random")

            self.entropys = self.random_protocol.exchange(out_bytes=True)
            self.random_num = int.from_bytes(self.entropys[:16], "big")
        else:
            self.random_num = 1

        self.dual_channel = {
            "min_max": DualChannel(name="min_max_" + FedConfig.node_id,
                                   ids=[FedConfig.get_assist_trainer()] + [FedConfig.node_id])}
        # send bins to assist_trainer
        self.dual_channel["min_max"].send(self.config.train_params["binning"])
        # init input bin_map
        self.bin_map = list()
        # init aggregation channel
        self.fedagg_executor = get_aggregation_leaf_inst(sec_conf=self.encryption)

    def init_data(self):
        if len(self.config.input_trainset) == 0:
            return None
        conf = self.config.input_trainset[0]
        if conf["type"] == "csv":
            path = os.path.join(conf['path'], conf['name'])
            has_label = conf["has_label"]
            has_id = conf['has_id']
            return CsvReader(path, has_id, has_label)
        else:
            return None

    def map_bin(self, x, split_point):
        bin_map = list(range(1, len(split_point) + 1))
        split_tile = np.tile(split_point, (len(x), 1))
        index = np.sum(x.to_numpy().reshape(-1, 1) - split_tile > 0, 1)
        self.bin_map.append([bin_map[i] for i in index])

    def bin_group(self, col_name):
        data_bin_y = pd.DataFrame(self.bin_map[col_name], columns=[col_name]).join(self.y)
        tmp_count = data_bin_y.groupby([col_name])['y'].agg({'count', 'sum'})
        pos_bin_count = tmp_count['sum']
        neg_bin_count = tmp_count['count'] - tmp_count['sum']
        pos_bin_count.name = "pos"
        neg_bin_count.name = "neg"
        # transform initial group result to the same length
        tmp_fill = pd.DataFrame(index=list(range(1, self.config.train_params["binning"]["bins"] + 1)))
        pos_bin_count = tmp_fill.join(pos_bin_count).fillna(0)
        neg_bin_count = tmp_fill.join(neg_bin_count).fillna(0)
        return [pos_bin_count, neg_bin_count]

    def fit(self):
        # calculate local min and max
        logger.info("Calculate local min and max of initial data")
        local_min = self.data.features().min(axis=0)
        local_max = self.data.features().max(axis=0)
        enc_local_min = local_min * abs(self.random_num)
        enc_local_max = local_max * abs(self.random_num)
        logger.info("Send local min and max to assist_trainer")
        self.dual_channel["min_max"].send((FedConfig.node_id, {"min": enc_local_min, "max": enc_local_max}))

        # receive signal from assist_trainer
        logger.info("Receive signal from assist_trainer")
        signal = self.dual_channel["min_max"].recv()
        min_sig = signal["min"]
        max_sig = signal["max"]
        min_send = [local_min[ind] if i else 0 for ind, i in enumerate(min_sig)]
        max_send = [local_max[ind] if j else 0 for ind, j in enumerate(max_sig)]

        # send final min and max to assist_trainer
        logger.info("Send final min and max to assist_trainer")
        self.dual_channel["min_max"].send({"min": min_send, "max": max_send})

        # receive split points from assist_trainer
        logger.info("Receive split points from assist_trainer")
        split_points = self.dual_channel["min_max"].recv()

        # map input into bins
        logger.info("Map raw data to bins")
        data_df = pd.DataFrame(self.data.features())
        map_tmp = list(range(len(split_points)))
        pd.Series(map_tmp).apply(lambda x: self.map_bin(data_df[x], split_points[x][1:]))

        # calculate pos_num and neg_num
        logger.info("Calculate pos_num and neg_num")
        data_bins_df = pd.DataFrame(self.bin_map).T
        pos_neg_bin = list()
        pd.Series(data_bins_df.columns).apply(lambda x: pos_neg_bin.append(self.bin_group(x)))
        pos_bin = [np.array(i[0]) for i in pos_neg_bin]
        neg_bin = [np.array(i[1]) for i in pos_neg_bin]

        # send pos_neg_bin to assist_trainer
        pos_bin_dict = OrderedDict(zip(range(len(pos_bin)), pos_bin))
        neg_bin_dict = OrderedDict(zip(range(len(neg_bin)), neg_bin))
        self.fedagg_executor.upload(pos_bin_dict, self.aggregation["weight"])
        self.fedagg_executor.upload(neg_bin_dict, self.aggregation["weight"])
