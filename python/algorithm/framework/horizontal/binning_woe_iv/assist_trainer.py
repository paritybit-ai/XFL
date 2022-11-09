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
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm.core.horizontal.aggregation.api import get_aggregation_root_inst
from common.communication.gRPC.python.channel import DualChannel
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger

from service.fed_config import FedConfig


def equal_width(min_, max_, bins):
    if min_ == max_:  # adjust end points before binning
        min_ -= .001 * abs(min_) if min_ != 0 else .001
        max_ += .001 * abs(max_) if max_ != 0 else .001
        bins = np.linspace(min_, max_, bins + 1, endpoint=True)
    else:  # adjust end points after binning
        bins = np.linspace(min_, max_, bins + 1, endpoint=True)
        adj = (max_ - min_) * 0.001  # 0.1% of the range
        bins[0] -= adj
    return list(bins)


class HorizontalBinningWoeIvAssistTrainer(object):
    def __init__(self, train_conf: dict):
        self.config = TrainConfigParser(train_conf)
        self.aggregation = self.config.train_params.get("aggregation", {})
        self.encryption = self.aggregation.get("encryption")
        self.nodes = FedConfig.get_label_trainer() + FedConfig.get_trainer()
        self.node_self = FedConfig.get_assist_trainer()
        self.dual_channel = {"min_max": {}, "woe": {}, "iv": {}}
        for node in self.nodes:
            self.dual_channel["min_max"][node] = DualChannel(name="min_max_" + node, ids=[self.node_self] + [node])
        tmp_lst = []
        for node in self.nodes:
            tmp_lst.append(self.dual_channel["min_max"][node].recv())
        self.binning = tmp_lst[0]
        self.fedagg_executor = get_aggregation_root_inst(sec_conf=self.encryption)

    def fit(self):
        # compare local min and max
        logger.info("Receive local min and max from trainers")
        node_lst = []
        min_lst = []
        max_lst = []
        for node in self.nodes:
            tmp = self.dual_channel["min_max"][node].recv()
            node_lst.append(tmp[0])
            min_lst.append(tmp[1]["min"])
            max_lst.append(tmp[1]["max"])
        logger.info("Compare min and max of all trainers")
        index_min = np.argmin(np.array(min_lst), axis=0)
        index_max = np.argmax(np.array(max_lst), axis=0)
        node_min = np.array([node_lst[i] for i in index_min])
        node_max = np.array([node_lst[j] for j in index_max])

        # send back min and max signal to trainers
        logger.info("Send back signal to all trainers")
        for node in self.nodes:
            min_msg = np.where(node_min == node, True, False)
            max_msg = np.where(node_max == node, True, False)
            self.dual_channel["min_max"][node].send({"min": min_msg, "max": max_msg})

        # receive final min and max from all trainers
        logger.info("Receive final min and max from all trainers")
        min_final = []
        max_final = []
        for node in self.nodes:
            final_rest = self.dual_channel["min_max"][node].recv()
            min_final.append(final_rest["min"])
            max_final.append(final_rest["max"])
        final_min = np.sum(min_final, axis=0)
        final_max = np.sum(max_final, axis=0)

        # split points
        split_points = []
        if self.binning["method"] == "equal_width":
            logger.info("Calculate split points when method is equal_width")
            for ind in range(len(final_min)):
                split_points.append(equal_width(final_min[ind], final_max[ind], self.binning["bins"]))
        logger.info("Send split points to trainers")
        for node in self.nodes:
            self.dual_channel["min_max"][node].send(split_points)

        # receive pos_num and neg_num from trainers and calculate pos and neg ratios
        logger.info("Receive pos_num and neg_num")
        pos_aggr = self.fedagg_executor.aggregate(average=False)
        neg_aggr = self.fedagg_executor.aggregate(average=False)

        # calculate total pos and neg
        logger.info("Calculate total pos_num and neg_num")
        pos_total = reduce(lambda x, y: x + y, list(pos_aggr.values())).sum() / len(pos_aggr)
        neg_total = reduce(lambda x, y: x + y, list(neg_aggr.values())).sum() / len(neg_aggr)
        assert pos_aggr[0].sum() == pos_total
        assert neg_aggr[0].sum() == neg_total

        # calculate woe for features
        woe_final = {}
        iv_final = {}
        for key in pos_aggr:
            logger.info("Calculate pos_prob and neg_prob")
            pos_prob = pos_aggr[key] / pos_total
            neg_prob = neg_aggr[key] / neg_total
            pos_prob = pd.Series(np.array(pos_prob).flatten()).apply(lambda x: 1e-7 if x == 0 else x)
            neg_prob = pd.Series(np.array(neg_prob).flatten()).apply(lambda x: 1e-7 if x == 0 else x)
            pos_aggr[key] = pos_prob
            neg_aggr[key] = neg_prob
            logger.info("Calculate woe")
            woe_pre = pos_prob / neg_prob
            woe = woe_pre.apply(lambda x: float("%.6f" % math.log(x)))
            woe_final[key] = woe.to_dict()
            logger.info("Calculate iv")
            iv_final[key] = float("%.6f" % np.sum((pos_prob - neg_prob) * woe))

        # save results
        logger.info("Save results")
        save_dir = self.config.output["path"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        guest_file_path = Path(save_dir)/self.config.output["result"]["name"]
        with open(guest_file_path, "w") as wf:
            json.dump({"woe": woe_final, "iv": iv_final, "split_points": dict(zip(range(len(split_points)),
                                                                                  split_points))}, wf)
