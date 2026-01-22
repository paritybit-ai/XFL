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
import pickle
import numpy as np
import pandas as pd
import pytest

from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.framework.horizontal.logistic_regression.assist_trainer import HorizontalLogisticRegressionAssistTrainer
from algorithm.framework.horizontal.logistic_regression.label_trainer import HorizontalLogisticRegressionLabelTrainer
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from common.communication.gRPC.python.channel import DualChannel
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
    case_df['y'] = np.where(case_df['x0'] + case_df['x2'] + case_df['x3'] > 2.5, 1, 0)
    case_df = case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']]
    case_df.head(800).to_csv(
        "/tmp/xfl/dataset/unit_test/train_data.csv", index=True
    )
    case_df.tail(200).to_csv(
        "/tmp/xfl/dataset/unit_test/test_data.csv", index=True
    )


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_logistic_regression/assist_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["valset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_data.csv"
        conf["output"]["path"] = "/tmp/xfl/checkpoints/unit_test"
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_logistic_regression/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.csv"
        conf["output"]["path"] = "/tmp/xfl/checkpoints/unit_test"
    yield conf


@pytest.fixture(scope="module", autouse=True)
def env():
    if not os.path.exists("/tmp/xfl/dataset/unit_test"):
        os.makedirs("/tmp/xfl/dataset/unit_test")
    if not os.path.exists("/tmp/xfl/checkpoints/unit_test"):
        os.makedirs("/tmp/xfl/checkpoints/unit_test")
    prepare_data()
    yield
    if os.path.exists("/tmp/xfl/dataset/unit_test"):
        shutil.rmtree("/tmp/xfl/dataset/unit_test")
    if os.path.exists("/tmp/xfl/checkpoints/unit_test"):
        shutil.rmtree("/tmp/xfl/checkpoints/unit_test")


class TestAggregation:
    @pytest.mark.parametrize("aggregation_method", ["fedavg", "fedprox", "scaffold"])
    def test_trainer(self, get_trainer_conf, get_assist_trainer_conf, aggregation_method, mocker):
        fed_method = None
        fed_assist_method = None
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        assist_conf["model_info"]["config"]["input_dim"] = 5
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        assist_conf["train_info"]["train_params"]["encryption"] = {"plain": {}}
        sec_conf = assist_conf["train_info"]["train_params"]["encryption"]["plain"]
        fed_method = AggregationPlainLeaf(sec_conf)
        fed_assist_method = AggregationPlainRoot(sec_conf)
        
        if aggregation_method == "fedprox":
            assist_conf["train_info"]["train_params"]["aggregation"] = {
                "method": {"fedprox": {"mu": 0.01}}
            }
        elif aggregation_method == "scaffold":
            assist_conf["train_info"]["train_params"]["aggregation"] = {
                "method": {"scaffold": {}}
            }
        else:
            assist_conf["train_info"]["train_params"]["aggregation"] = {
                "method": {"fedavg": {}}
            }

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
        
        lrt = HorizontalLogisticRegressionLabelTrainer(conf)
        lrt_a = HorizontalLogisticRegressionAssistTrainer(assist_conf)
        esflag_recv = pickle.dumps(False) + EOV
        params_plain_recv = pickle.dumps(lrt_a.model.state_dict()) + EOV
        params_send = fed_method._calc_upload_value(lrt.model.state_dict(), len(lrt.train_dataloader.dataset))
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
        mocker.patch("service.fed_control._send_progress")
        lrt.fit()
        lrt_a.fit()
