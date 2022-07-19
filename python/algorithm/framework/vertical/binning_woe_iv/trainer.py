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


import json
import os
import time

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool

from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.logger import logger
from ..pearson.base import VerticalPearsonBase
from .base import VerticalBinningWoeIvBase


class VerticalBinningWoeIvTrainer(VerticalBinningWoeIvBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """[summary]

        Args:
            train_conf (dict): [description]
            model_conf (dict, optional): [description]. Defaults to None.
        """
        super().__init__(train_conf, label=False, *args, **kwargs)
        self.encrypt_id_label_pair = None
        # self.pool = ProcessPool(self.extra_config["poolNum"])
        self.bins_count = {}
        self.woe_feedback_list = {}
        self.broadcast_channel = BroadcastChannel(name="vertical_binning_woe_iv_channel")
        self.feature_mapping = VerticalPearsonBase.string_encryption(self.df.columns)
        self.df.columns = list(pd.Series(self.df.columns).apply(lambda x: self.feature_mapping[x]))
        self.woe_map = dict(
            zip(pd.Series(self.woe_map.keys()).apply(lambda x: self.feature_mapping[x]), self.woe_map.values()))
        logger.info("node-2:successfully binning.")

    def woe_pre(self, feat):
        feature_df = self.encrypt_id_label_pair.join(self.df[feat])
        tmp = feature_df.groupby([feat])['y'].agg({'count', 'sum'})
        return tmp

    def fit(self):

        encryption_config = self.train_params["encryption_params"]
        encryption_method = encryption_config["method"].lower()

        #
        if encryption_method == "paillier":
            # id = self.df.index.tolist()
            pub_context = self.broadcast_channel.recv(use_pickle=False)
            pub_context = Paillier.context_from(pub_context)
            en_label = self.broadcast_channel.recv(use_pickle=False)
            en_label = Paillier.ciphertext_from(pub_context, en_label)

            # if len(list(id)) != len(en_label):
            #     raise IndexError(f"Table size not match. Local table size is {len(list(id))}, "
            #                      f"incoming table size is {len(en_label)}")

            self.encrypt_id_label_pair = pd.DataFrame(en_label).rename(columns={0: 'y'})
            self.encrypt_id_label_pair.index = self.df.index

            print("Start count bins for trainer")

            time_s = time.time()

            # tmp = self.pool.map(self.woe_pre, list(self.df.columns))
            def woe_pre_plus(batch_data):
                _tmp = pd.DataFrame(columns=['y', 'col_value', 'col_name'],
                                    index=range(len(batch_data[0]) * batch_size))
                for _id in range(len(batch_data)):
                    col = batch_data[_id].columns[0]
                    batch_data[_id]['col_name'] = col
                    batch_data[_id] = batch_data[_id].rename(columns={col: 'col_value'})
                    batch_data[_id] = self.encrypt_id_label_pair.join(batch_data[_id])
                    batch_data[_id].index = range(len(batch_data[0]) * _id, len(batch_data[0]) * (_id + 1))
                    _tmp.loc[len(batch_data[0]) * _id:len(batch_data[0]) * (_id + 1), :] = batch_data[_id]
                    # tmp = pd.concat([tmp, feat])
                tmp_gp = _tmp.groupby(['col_name', 'col_value'])['y'].agg({'count', 'sum'})
                del _tmp, batch_data
                bins_count = dict(zip(tmp_gp.index.levels[0], [tmp_gp.loc[ii]['count']
                                                               for ii in tmp_gp.index.levels[0]]))
                woe_feedback_list = dict(
                    zip(tmp_gp.index.levels[0], [tmp_gp.loc[ii]['sum'] for ii in tmp_gp.index.levels[0]]))
                return bins_count, woe_feedback_list

            data_batch = []
            col_name = list(self.df.columns)
            batch_size = 30
            div = int(np.ceil(len(col_name) / batch_size))
            for i in range(div):
                if i == div - 1:
                    num_lst = list(range(i * batch_size, len(col_name)))
                    t = [pd.DataFrame(self.df[col_name[val]]) for val in num_lst]
                else:
                    num_lst = list(range(i * batch_size, (i + 1) * batch_size))
                    t = [pd.DataFrame(self.df[col_name[val]]) for val in num_lst]
                data_batch.append(t)
            del self.df
            woe_pre_plus([pd.DataFrame([1, 2, 3])])  # improve coverage
            with ProcessPool(self.train_params["pool_num"]) as pool:
                tmp = pool.map(woe_pre_plus, data_batch)
            for i in tmp:
                self.bins_count.update(i[0])
                self.woe_feedback_list.update(i[1])
            print("Trainer sum costs:" + str(time.time() - time_s))
        elif encryption_method == "plain":
            self.encrypt_id_label_pair = self.broadcast_channel.recv(use_pickle=True)
            self.encrypt_id_label_pair = pd.DataFrame(self.encrypt_id_label_pair)
            tmp = []
            print("Start count bins for trainer")
            time_s = time.time()
            pd.Series(self.df.columns).apply(lambda x: tmp.append(self.woe_pre(x)))
            self.bins_count = dict(zip(self.df.columns, [i['count'] for i in tmp]))
            self.woe_feedback_list = dict(zip(self.df.columns, [i['sum'] for i in tmp]))
            print("Trainer sum costs:" + str(time.time() - time_s))
        # else:
        #     raise ValueError(
        #         f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'plain'.")

        # woe name map
        def woe_name_map(feat):
            self.woe_feedback_list[feat].index = pd.Series(self.woe_feedback_list[feat].index).apply(
                lambda x: self.woe_map[feat][x])
            self.bins_count[feat].index = pd.Series(self.bins_count[feat].index).apply(lambda x: self.woe_map[feat][x])
        pd.Series(self.woe_feedback_list.keys()).apply(lambda x: woe_name_map(x))

        if encryption_method == "paillier":
            for id_, feature in self.woe_feedback_list.items():
                self.woe_feedback_list[id_] = feature.apply(lambda x: x.serialize())

        send_msg = {"woe_feedback_list": self.woe_feedback_list, "bins_count": self.bins_count}
        self.broadcast_channel.send(send_msg)
        # save feature map
        save_dir = self.output["trainset"]["path"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        host_file_path = f'{save_dir}/{self.output["trainset"]["name"]}_feature_mapping.json'
        with open(host_file_path, "w") as wf:
            json.dump({"feature_mapping": self.feature_mapping}, wf)
