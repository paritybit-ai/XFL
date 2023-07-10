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
from scipy.stats import normaltest
import pickle
import pandas as pd
import torch
import pytest

from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from algorithm.framework.horizontal.nbafl.assist_trainer import HorizontalNbaflAssistTrainer
from algorithm.framework.horizontal.nbafl.label_trainer import HorizontalNbaflLabelTrainer
from common.communication.gRPC.python.channel import DualChannel
from common.utils.logger import logger
from common.utils.config_sync import ConfigSynchronizer
from common.communication.gRPC.python.commu import Commu

MOV = b"@" # middle of value
EOV = b"&" # end of value

def prepare_data():
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': [0] * 1000,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = np.where(
        case_df['x0'] + case_df['x2'] + case_df['x3'] > 2.5, 1, 0)
    case_df = case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']]
    case_df.head(800).to_csv(
        "/opt/dataset/unit_test/train_data.csv", index=True
    )
    case_df.tail(200).to_csv(
        "/opt/dataset/unit_test/test_data.csv", index=True
    )


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_nbafl/assist_trainer.json") as f:
        conf = json.load(f)
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_nbafl/trainer.json") as f:
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


class TestNbafl:

    def test_uplink_sigma(self, get_trainer_conf, get_assist_trainer_conf, mocker):
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        recv_mocker = mocker.patch.object(
            DualChannel, "recv", 
            return_value = {
                "model_info":assist_conf["model_info"], "train_info": assist_conf["train_info"]
            }
        )
        nbafl_t = HorizontalNbaflLabelTrainer(conf)
        logger.info(f"{len(nbafl_t.train_dataloader.dataset)} of data")
        nbafl_t._calc_uplink_sigma({})
        sigma_u = nbafl_t.sigma_u
        expected_sigma_u = np.sqrt(2 * np.log(12.5)) / 80
        logger.info(f"expected uplink sigma: {expected_sigma_u}")
        assert np.abs(sigma_u - expected_sigma_u) < 0.0001

    def test_uplink_add_noise(self, get_trainer_conf, get_assist_trainer_conf, mocker):
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        mocker.patch.object(
            DualChannel, "recv", 
            return_value = {
                "model_info":assist_conf["model_info"], "train_info": assist_conf["train_info"]
            }
        )
        nbafl_t = HorizontalNbaflLabelTrainer(conf)
        nbafl_t.sigma_u = 0.1
        diff_list = []
        orig_params = [
            param.data.detach().clone() for param in nbafl_t.model.parameters()
        ]
        np.random.seed(42)
        torch.manual_seed(42)
        for _ in range(3):
            iter_diff_list = []
            nbafl_t._add_noise({})
            for orig_param, new_param in zip(orig_params, nbafl_t.model.parameters()):
                iter_diff_list.extend(torch.flatten(
                    orig_param - new_param.data.detach()
                ).numpy().tolist())
            diff_list.extend(iter_diff_list)
        _, pval = normaltest(diff_list)
        logger.info("Normal test p-value: {}".format(pval))
        assert pval > 0.1
        diff_sigma = np.std(diff_list)
        logger.info("Diff std: {}".format(diff_sigma))
        assert np.abs(diff_sigma - nbafl_t.sigma_u) < 0.05

    def test_downlink_sigma(self, get_trainer_conf, get_assist_trainer_conf, mocker):
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        mocker.patch.object(
            DualChannel, "recv", 
            return_value = {
                "model_info":assist_conf["model_info"], "train_info": assist_conf["train_info"]
            }
        )
        nbafl_at = HorizontalNbaflAssistTrainer(conf)
        nbafl_at.min_sample_num = 10
        expected_sigma_d = 10 * \
            np.sqrt(2 * np.log(12.5)) * np.sqrt((25-8) / 20)
        nbafl_at._calc_downlink_sigma({})
        assert (nbafl_at.sigma_d - expected_sigma_d) < 0.0001

    def test_label_trainer(self, get_trainer_conf, get_assist_trainer_conf, mocker):
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        mocker.patch.object(
            DualChannel, "recv", 
            return_value = {
                "model_info":assist_conf["model_info"], "train_info": assist_conf["train_info"]
            }
        )

        nbafl_t = HorizontalNbaflLabelTrainer(conf)
        nbafl_t.sigma_u = 0.1
        mocker.patch.object(
            ConfigSynchronizer, "__init__", return_value=None
        )
        mocker.patch.object(
            ConfigSynchronizer, "sync", return_value=assist_conf
        )
        mocker.patch("service.fed_control._send_progress")
        nbafl_at = HorizontalNbaflAssistTrainer(assist_conf)
        
        sec_conf = assist_conf["train_info"]["train_params"]["encryption"]["plain"]
        fed_method = AggregationPlainLeaf(sec_conf)
        fed_assist_method = AggregationPlainRoot(sec_conf)
        esflag_recv = pickle.dumps(False) + EOV
        params_plain_recv = pickle.dumps(nbafl_at.model.state_dict()) + EOV
        params_send = fed_method._calc_upload_value(nbafl_t.model.state_dict(), len(nbafl_t.train_dataloader.dataset))
        params_collect = pickle.dumps(params_send)
        agg_otp = fed_assist_method._calc_aggregated_params(list(map(lambda x: pickle.loads(x), [params_collect,params_collect])))
        
        def mock_recv(*args, **kwargs):
            if recv_mocker.call_count % 2 == 1:
                return esflag_recv
            else:
                return params_plain_recv
        
        def mock_agg(*args, **kwargs):
            return agg_otp

        recv_mocker = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_recv
        )
        mocker.patch.object(
            AggregationPlainRoot, "aggregate", side_effect=mock_agg
        )
        nbafl_t.fit()
        nbafl_at.min_sample_num = 10
        mocker.patch.object(
            DualChannel, "recv", return_value=10
        )
        nbafl_at.fit()
