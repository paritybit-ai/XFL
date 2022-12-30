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
import shutil
from typing import OrderedDict

import torch

from algorithm.framework.horizontal.binning_woe_iv.assist_trainer import \
    HorizontalBinningWoeIvAssistTrainer
from algorithm.framework.horizontal.binning_woe_iv.label_trainer import HorizontalBinningWoeIvLabelTrainer

import numpy as np
import pandas as pd
import pytest

import service.fed_config
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu

from common.crypto.key_agreement.diffie_hellman import DiffieHellman


def map_bin(x, split_point):
    bin_map = list(range(1, len(split_point) + 1))
    split_tile = np.tile(split_point, (len(x), 1))
    index = np.sum(x.to_numpy().reshape(-1, 1) - split_tile > 0, 1)
    return [bin_map[i] for i in index]


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


def prepare_data():
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': [0] * 1000,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = np.where(case_df['x0'] + case_df['x2'] + case_df['x3'] > 2.5, 1, 0)
    case_df = case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']]
    case_df.head(800).to_csv(
        "/opt/dataset/unit_test/train_data.csv", index=True
    )
    case_df.tail(200).to_csv(
        "/opt/dataset/unit_test/test_data.csv", index=True
    )


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/horizontal_binning_woe_iv/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.csv"
    yield conf


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_binning_woe_iv/assist_trainer.json") as f:
        conf_a = json.load(f)
        conf_a["output"]["path"] = "/opt/checkpoints/unit_test1"
    yield conf_a


@pytest.fixture(scope="module", autouse=True)
def env():
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    prepare_data()
    yield
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test1"):
        shutil.rmtree("/opt/checkpoints/unit_test1")


class TestLogisticRegression:
    @pytest.mark.parametrize("encryption_method", ["plain", "otp"])
    def test_trainer(self, get_label_trainer_conf, get_assist_trainer_conf, encryption_method, mocker):
        conf = get_label_trainer_conf
        conf_a = get_assist_trainer_conf
        conf["train_info"]["train_params"]["aggregation"]["encryption"]["method"] = encryption_method
        # mock init
        if encryption_method == "plain":
            mocker.patch.object(
                DualChannel, "__init__", return_value=None
            )
        elif encryption_method == "otp":
            pass
        mocker.patch.object(Commu, "trainer_ids", ['node-1', 'node-2'])
        mocker.patch.object(Commu, "scheduler_id", "assist_trainer")
        mocker.patch.object(Commu, "node_id", "node-1")
        mocker.patch.object(service.fed_config.FedConfig, "node_id", "node-1")
        mocker.patch.object(
            DualChannel, "send", return_value=0
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_assist_trainer", return_value="assist_trainer"
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=[]
        )
        mocker.patch.object(
            DiffieHellman, "exchange", return_value=bytes(30000)
        )
        # init label_trainer
        h_woe_iv = HorizontalBinningWoeIvLabelTrainer(conf)
        local_min = h_woe_iv.data.features().min(axis=0)
        local_max = h_woe_iv.data.features().max(axis=0)
        tmp = ("node-1", {"min": local_min, "max": local_max})
        node_lst = [tmp[0]]
        min_lst = [tmp[1]["min"]]
        max_lst = [tmp[1]["max"]]
        index_min = np.argmin(np.array(min_lst), axis=0)
        index_max = np.argmax(np.array(max_lst), axis=0)
        node_min = np.array([node_lst[i] for i in index_min])
        node_max = np.array([node_lst[j] for j in index_max])
        min_msg = np.where(node_min == "node-1", True, False)
        max_msg = np.where(node_max == "node-1", True, False)

        min_send = [local_min[ind] if i else 0 for ind, i in enumerate(min_msg)]
        max_send = [local_max[ind] if j else 0 for ind, j in enumerate(max_msg)]

        min_final = [min_send]
        max_final = [max_send]
        final_min = np.sum(min_final, axis=0)
        final_max = np.sum(max_final, axis=0)

        split_points = []
        if conf["train_info"]["train_params"]["binning"]["method"] == "equal_width":
            for ind in range(len(final_min)):
                split_points.append(equal_width(final_min[ind], final_max[ind],
                                                conf["train_info"]["train_params"]["binning"]["bins"]))

        def mock_recv_dual(*args, **kwargs):
            if mock_label_recv_dual.call_count == 1:
                return {"min": min_msg, "max": max_msg}
            elif mock_label_recv_dual.call_count == 2:
                return split_points

        mock_label_recv_dual = mocker.patch.object(
            h_woe_iv.dual_channel["min_max"], "recv", side_effect=mock_recv_dual
        )
        mocker.patch.object(
            h_woe_iv.fedagg_executor, "upload", return_value=0
        )

        # fit label_trainer
        h_woe_iv.fit()

        # mock assist_trainer
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=['node-1']
        )
        mocker.patch.object(
            DualChannel, "recv", return_value={
                "method": "equal_width",
                "bins": 5
            }
        )
        h_woe_iv_assist = HorizontalBinningWoeIvAssistTrainer(conf_a)

        # mock assist_trainer
        def mock_recv_dual_a(*args, **kwargs):
            if mock_assist_recv_dual.call_count == 1:
                return ("node-1", {"min": local_min, "max": local_max})
            elif mock_assist_recv_dual.call_count == 2:
                return {"min": min_send, "max": max_send}

        mock_assist_recv_dual = mocker.patch.object(
            h_woe_iv_assist.dual_channel["min_max"]["node-1"], "recv", side_effect=mock_recv_dual_a
        )

        def bin_group(col_name, y):
            data_bin_y = pd.DataFrame(bin_map[col_name], columns=[col_name]).join(y)
            tmp_count = data_bin_y.groupby([col_name])['y'].agg({'count', 'sum'})
            pos_bin_count = tmp_count['sum']
            neg_bin_count = tmp_count['count'] - tmp_count['sum']
            pos_bin_count.name = "pos"
            neg_bin_count.name = "neg"
            # transform initial group result to the same length
            tmp_fill = pd.DataFrame(index=list(range(1, h_woe_iv.config.train_params["binning"]["bins"] + 1)))
            pos_bin_count = tmp_fill.join(pos_bin_count).fillna(0)
            neg_bin_count = tmp_fill.join(neg_bin_count).fillna(0)
            return [pos_bin_count, neg_bin_count]

        bin_map = list()
        data_df = pd.DataFrame(h_woe_iv.data.features())
        map_tmp = list(range(len(split_points)))
        pd.Series(map_tmp).apply(lambda x: bin_map.append(map_bin(data_df[x], split_points[x][1:])))
        data_bins_df = pd.DataFrame(bin_map).T
        pos_neg_bin = list()
        pd.Series(data_bins_df.columns).apply(lambda x: pos_neg_bin.append(bin_group(x, h_woe_iv.y)))
        pos_bin = [np.array(i[0]) for i in pos_neg_bin]
        neg_bin = [np.array(i[1]) for i in pos_neg_bin]
        pos_bin_dict = OrderedDict(zip(range(len(pos_bin)), pos_bin))
        neg_bin_dict = OrderedDict(zip(range(len(neg_bin)), neg_bin))

        def mock_aggregation(*args, **kwargs):
            return_value = OrderedDict(zip(range(len(pos_bin)), np.zeros(len(pos_bin))))
            if mock_aggregation_a.call_count == 1:
                for key in pos_bin_dict.keys():
                    return_value[key] += pos_bin_dict[key]
            elif mock_aggregation_a.call_count == 2:
                for key in neg_bin_dict.keys():
                    return_value[key] += neg_bin_dict[key]
            return return_value

        mock_aggregation_a = mocker.patch.object(
            h_woe_iv_assist.fedagg_executor, "aggregate", side_effect=mock_aggregation
        )
        # fit assist_trainer
        h_woe_iv_assist.fit()

        assert os.path.exists("/opt/checkpoints/unit_test1/woe_iv_result_[STAGE_ID].json")
        with open("/opt/checkpoints/unit_test1/woe_iv_result_[STAGE_ID].json", "r",
                  encoding='utf-8') as f:
            conf = json.loads(f.read())
            for k in ["woe", "iv", "split_points"]:
                assert k in conf
                assert len(conf[k]) == np.shape(h_woe_iv.data.features())[1]
