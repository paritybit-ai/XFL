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
import math
import os
import time

import numpy as np
import pandas as pd

from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier, PaillierCiphertext
from common.utils.logger import logger
from .base import VerticalBinningWoeIvBase


class VerticalBinningWoeIvLabelTrainer(VerticalBinningWoeIvBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """
        Args:
            train_conf: training parameters
            *args:
            **kwargs:
        """
        super().__init__(train_conf, label=True, *args, **kwargs)
        self.neg_total_count, self.pos_total_count = 0, 0
        logger.info("node-1:successfully binning.")
        self.woe_dict_total = {}
        self.iv_dict_total = {}
        self.neg_bin_count = {}
        self.pos_bin_count = {}
        self.neg_bin_ratio = {}
        self.pos_bin_ratio = {}
        self.broadcast_channel = BroadcastChannel(name="vertical_binning_woe_iv_channel")

    def fit(self):
        # broadcast_channel = BroadcastChannel(name="vertical_binning_woe_iv_channel")

        encryption_config = self.train_params["encryption_params"]
        encryption_method = encryption_config["method"].lower()

        # if encryption_method == "paillier":
        #     pri_context = Paillier.context(encryption_config["key_bit_size"], djn_on=encryption_config["djn_on"])
        #     self.broadcast_channel.broadcast(pri_context.to_public().serialize(), use_pickle=False)
        # elif encryption_method == "plain":
        #     pass
        # else:
        #     raise ValueError(
        #         f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'plain'.")

        # logger.info("Start calculate host IV with WOE values.")
        self.label_trainer_woe_iv()

        if encryption_method == "paillier":
            pri_context = Paillier.context(encryption_config["key_bit_size"], djn_on=encryption_config["djn_on"])
            self.broadcast_channel.broadcast(pri_context.to_public().serialize(), use_pickle=False)
            num_cores = -1 if encryption_config["parallelize_on"] else 1
            label = self.df[["y"]].to_numpy().flatten().astype(np.int32)
            logger.info(f"Encrypting label using {encryption_method} method.")
            en_label = Paillier.encrypt(pri_context,
                                        label,
                                        precision=encryption_config["precision"],
                                        obfuscation=True,
                                        num_cores=num_cores)
            logger.info("Encryption complete.")

            self.broadcast_channel.broadcast(Paillier.serialize(en_label), use_pickle=False)
        elif encryption_method == "plain":
            id_label_pair = self.df[["y"]]
            self.broadcast_channel.broadcast(id_label_pair, use_pickle=True)

        feedback_list = self.broadcast_channel.collect()
        assert len(self.broadcast_channel.remote_ids) == len(feedback_list)
        for uid, feedback in zip(self.broadcast_channel.remote_ids, feedback_list):
            client_woe_dict, client_iv_dict = {}, {}
            if encryption_method == "paillier":
                logger.info(f"Decrypting woe_feedback using {encryption_method} method.")
                for _id, feature in feedback["woe_feedback_list"].items():
                    c = feature.apply(lambda x: PaillierCiphertext.deserialize_from(pri_context, x))
                    feedback["woe_feedback_list"][_id] = c.apply(lambda x: Paillier.decrypt(pri_context,
                                                                                            x,
                                                                                            dtype='float',
                                                                                            num_cores=num_cores))
                logger.info("Decryption Complete.")

            woe_feedback_list, bins_count = feedback["woe_feedback_list"], feedback["bins_count"]

            logger.info("Start calculate woe for trainer")
            time_s = time.time()
            for k, v in woe_feedback_list.items():
                # featName = "{}_{}".format(uid, k)

                client_woe_dict[k], client_iv_dict[k] = {}, 0
                neg_ = bins_count[k] - v
                pos_prob, neg_prob = (v / self.pos_total_count), (neg_ / self.neg_total_count)
                pos_prob = pos_prob.apply(lambda x: 1e-7 if x == 0 else x)
                neg_prob = neg_prob.apply(lambda x: 1e-7 if x == 0 else x)
                woe_pre = pos_prob / neg_prob
                woe = woe_pre.apply(lambda x: float("%.6f" % math.log(x)))
                # woe.index = pd.Series(woe.index).apply(lambda x: int(x))

                # v.index = pd.Series(v.index).apply(lambda x: int(x))
                # neg_.index = pd.Series(neg_.index).apply(lambda x: int(x))
                self.pos_bin_count[k] = v.to_dict()
                self.neg_bin_count[k] = neg_.to_dict()
                pos_prob = pos_prob.apply(lambda x: float("%.6f" % x))
                neg_prob = neg_prob.apply(lambda x: float("%.6f" % x))
                self.pos_bin_ratio[k] = pos_prob.to_dict()
                self.neg_bin_ratio[k] = neg_prob.to_dict()
                client_woe_dict[k] = woe.to_dict()
                client_iv_dict[k] += float("%.6f" % np.sum((pos_prob - neg_prob) * woe))
            logger.info("Trainer woe cost:" + str(time.time() - time_s))

            logger.info("Calculate host IV with WOE values completed.")
            # logger.info("Host WOE dictionary: {}".format(client_woe_dict))
            # logger.info("Host IV dictionary: {}".format(client_iv_dict))
            # Save host dicts
            self.woe_dict_total.update(client_woe_dict)
            self.iv_dict_total.update(client_iv_dict)
            save_dir = self.output["trainset"]["path"]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            guest_file_path = f'{save_dir}/{self.output["trainset"]["name"]}.json'
            with open(guest_file_path, "w") as wf:
                json.dump({"woe": self.woe_dict_total, "iv": self.iv_dict_total, "count_neg": self.neg_bin_count,
                           "count_pos": self.pos_bin_count, "ratio_pos": self.pos_bin_ratio,
                           "ratio_neg": self.neg_bin_ratio}, wf)
            logger.info("Host {} WOE & IV values saved as {}.".format(uid, guest_file_path))

    def label_trainer_woe_iv(self):
        logger.info("Start calculate Guest IV with WOE values.")
        woe_dict, iv_dict = {}, {}
        # count_neg_dict, count_pos_dict, bins_total, percentage, bad_rate = {}, {}, {}, {}, {}
        # good_percentage, bad_percentage = {}, {}
        # # count total label = 0, 1
        total_count = self.df.groupby("y")["y"].count()
        self.neg_total_count, self.pos_total_count = total_count[0], total_count[1]

        feat_woe = set(self.df.columns).difference(set("y"))

        logger.info("Start calculate woe for label trainer")
        time_s = time.time()
        for feature in feat_woe:
            woe_dict[feature], iv_dict[feature] = {}, 0
            feature_df = self.df[[feature, "y"]]

            tmp_count = feature_df.groupby([feature])['y'].agg({'count', 'sum'})
            neg_bin_count = tmp_count['count'] - tmp_count['sum']
            pos_prob, neg_prob = (tmp_count['sum'] / self.pos_total_count), (neg_bin_count / self.neg_total_count)
            pos_prob = pos_prob.apply(lambda x: 1e-7 if x == 0 else x)
            neg_prob = neg_prob.apply(lambda x: 1e-7 if x == 0 else x)
            woe_pre = pos_prob / neg_prob
            woe = woe_pre.apply(lambda x: float("%.6f" % math.log(x)))
            iv_dict[feature] = float("%.6f" % np.sum((pos_prob - neg_prob) * woe))

            woe.index = pd.Series(woe.index).apply(lambda x: self.woe_map[feature][x])
            tmp_count['sum'].index = pd.Series(tmp_count['sum'].index).apply(lambda x: self.woe_map[feature][x])
            neg_bin_count.index = pd.Series(neg_bin_count.index).apply(lambda x: self.woe_map[feature][x])
            self.pos_bin_count[feature] = tmp_count['sum'].to_dict()
            self.neg_bin_count[feature] = neg_bin_count.to_dict()
            pos_prob = pos_prob.apply(lambda x: float("%.6f" % x))
            neg_prob = neg_prob.apply(lambda x: float("%.6f" % x))
            pos_prob.index = pd.Series(pos_prob.index).apply(lambda x: self.woe_map[feature][x])
            neg_prob.index = pd.Series(neg_prob.index).apply(lambda x: self.woe_map[feature][x])
            self.pos_bin_ratio[feature] = pos_prob.to_dict()
            self.neg_bin_ratio[feature] = neg_prob.to_dict()
            woe_dict[feature] = woe.to_dict()

        logger.info("label trainer cost:" + str(time.time() - time_s))

        logger.info("Calculate Guest IV with WOE values completed.")
        # # logger.info("Guest WOE dictionary: {}".format(woe_dict))
        # # logger.info("Guest IV dictionary: {}".format(iv_dict))
        # # Save guest dicts
        # save_dir = self.output["dataset"]["path"]
        # if not os.path.exists(save_dir): os.makedirs(save_dir)
        # guest_file_path = f'{save_dir}/{self.output["dataset"]["name"]}.json'
        # with open(guest_file_path, "a") as wf:
        #     json.dump({"woe": woe_dict, "iv": iv_dict, "count_neg": count_neg_dict, "count_pos": count_pos_dict,
        #                "bins_total": bins_total, "percentage": percentage, "bad_rate": bad_rate,
        #                "good_percentage": good_percentage, "bad_percentage": bad_percentage}, wf)
        self.woe_dict_total.update(woe_dict)
        self.iv_dict_total.update(iv_dict)
        # logger.info("Guest WOE & IV values saved as {}.".format(guest_file_path))
        logger.info("Guest WOE & IV values saved")
