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
import numpy as np
import pandas as pd
import pytest
import torch

from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.framework.transfer.logistic_regression.label_trainer import \
    TransferLogisticRegressionLabelTrainer
from algorithm.framework.transfer.logistic_regression.trainer import \
    TransferLogisticRegressionTrainer
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu
from common.utils.config_sync import ConfigSynchronizer


def prepare_data():
    case_dict = {}
    for i in range(30):
        if i == 0:
            case_dict.update({f"x{i:0>2d}": [0] * 100})
        elif i == 10:
            case_dict.update({f"x{i:0>2d}": [1] * 100})
        elif i == 20:
            case_dict.update({f"x{i:0>2d}": [2] * 100})
        else:
            case_dict.update({f"x{i:0>2d}": np.random.random(100)})
    case_df = pd.DataFrame(case_dict)
    case_df["y"] = np.where(case_df["x00"] + case_df["x10"] + case_df["x20"] + case_df["x29"] > 3.5, 1, 0)
    columns_labeled = ["y"] + [f"x{i:0>2d}" for i in range(15)]
    columns_1 = [f"x{i:0>2d}" for i in range(15, 30)]
    case_df[columns_labeled].head(60).to_csv(
        "/opt/dataset/unit_test/train_labeled.csv", index=True
    )
    case_df[columns_labeled].tail(20).to_csv(
        "/opt/dataset/unit_test/test_labeled.csv", index=True
    )
    case_df[columns_1].head(80).tail(60).to_csv(
        "/opt/dataset/unit_test/train_1.csv", index=True
    )
    case_df[columns_1].tail(20).to_csv(
        "/opt/dataset/unit_test/test_1.csv", index=True
    )
    overlap_index = np.linspace(20, 59, 40, dtype=np.int16)
    np.save("/opt/dataset/unit_test/overlap_index.npy", overlap_index)


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/transfer_logistic_regression/label_trainer.json") as f:
        conf = json.load(f)
    yield conf

@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/transfer_logistic_regression/trainer.json") as f:
        conf = json.load(f)
    yield conf

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


class TestTransferLogisticRegression:
    @pytest.mark.parametrize("encryption_method", ["plain"])
    def test_label_trainer(self, get_label_trainer_conf, get_trainer_conf, encryption_method, mocker):
        # label trainer 流程测试
        conf = get_label_trainer_conf
        Commu.node_id = "node-1"
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'node-1'
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            FedConfig, "get_trainer", return_value=["node-2"]
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(DualChannel, "send", return_value=0)

        def mock_recv():
            if mock_channel_recv.call_count <= lrlt.global_epoch * lrlt.local_epoch:
                return (torch.rand(40, 5), torch.rand(40, 5, 5), torch.rand(40, 5))
            else:
                return torch.rand(20, 5)

        mock_channel_recv = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_recv
        )
        mocker.patch.object(
            ConfigSynchronizer, "__init__", return_value=None
        )
        mocker.patch.object(
            ConfigSynchronizer, "sync", return_value=conf
        )
        lrlt = TransferLogisticRegressionLabelTrainer(conf)
        lrlt.fit()

        # load pretrained model
        lrlt.pretrain_model_path = "/opt/checkpoints/unit_test"
        lrlt.pretrain_model_name = "transfer_logitstic_regression_0.model"
        lrlt._set_model()

    @pytest.mark.parametrize("encryption_method", ["plain"])
    def test_trainer(self, get_trainer_conf, get_label_trainer_conf, encryption_method, mocker):
        # trainer 流程测试
        conf = get_trainer_conf
        conf_l = get_label_trainer_conf
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'node-1'
        mocker.patch.object(Commu, "node_id", "node-1")
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "node_id", 'node-1'
        )
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        recv_mocker = mocker.patch.object(
            DualChannel, "recv", 
            return_value = {
                "model_info":conf_l["model_info"], "train_info": conf_l["train_info"]
            }
        )
        lrt = TransferLogisticRegressionTrainer(conf)
        mocker.patch.object(
            DualChannel, "send", return_value=0
        )
        mocker.patch.object(
            DualChannel, "recv", return_value=(torch.rand(40, 5, 5), torch.rand(40, 5), torch.rand(40, 1))
        )
        lrt.fit()

        # load pretrained model
        lrt.pretrain_model_path = "/opt/checkpoints/unit_test"
        lrt.pretrain_model_name = "transfer_logitstic_regression_0.model"
        lrt._set_model()
