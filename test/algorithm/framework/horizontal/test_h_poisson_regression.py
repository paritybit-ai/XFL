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
from random import SystemRandom
import pickle
import numpy as np
import pandas as pd
import pytest
import torch
from gmpy2 import powmod

from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.framework.horizontal.poisson_regression.assist_trainer import HorizontalPoissonRegressionAssistTrainer
from algorithm.framework.horizontal.poisson_regression.label_trainer import HorizontalPoissonRegressionLabelTrainer
from algorithm.core.horizontal.aggregation.aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.key_agreement.contants import primes_hex


MOV = b"@"  # middle of value
EOV = b"&"  # end of value


def prepare_data():
    np.random.seed(42)
    case_df = pd.DataFrame({
        'x0': np.random.random(1000) + 0.5,
        'x1': [0] * 1000,
        'x2': np.random.random(1000) + 1.0,
        'x3': np.random.random(1000),
        'x4': np.random.random(1000) + 1.0
    })

    feat_mat = case_df.values
    lin_theo = np.dot(feat_mat, np.array([1, 0, 1, 3, 0]))
    print(f"Max of lin_theo: {lin_theo.max()}")
    print(f"Min of lin_theo: {lin_theo.min()}")
    theore_pred = np.exp(np.dot(feat_mat, np.array([1, 0, 1, 3, 0])))
    print(f"Theoretical pred: {theore_pred}")
    print(f"Min theoretical pred: {theore_pred.min()}")
    print(f"Min of case_df: {case_df.min(axis=0)}")
    case_df['y'] = np.rint(
        np.exp(case_df['x0'] + 1*case_df['x2'] + 2*case_df['x3'])
    )
    case_df = case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']]
    case_df.head(800).to_csv(
        "/opt/dataset/unit_test/train_data.csv", index=False
    )
    case_df.tail(200).to_csv(
        "/opt/dataset/unit_test/test_data.csv", index=False
    )


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_poisson_regression/assist_trainer.json") as f:
        conf = json.load(f)
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_poisson_regression/trainer.json") as f:
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


class TestPoissonRegression:
    @pytest.mark.parametrize("encryption_method", ['plain', 'otp'])
    def test_trainer(self, get_trainer_conf, get_assist_trainer_conf, encryption_method, mocker):
        fed_method = None
        fed_assist_method = None
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(
            FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        mocker.patch.object(FedNode, "node_id", "node-1")
        if encryption_method == "plain":
            assist_conf["train_info"]["train_params"]["encryption"] = {"plain": {}}

            sec_conf = assist_conf["train_info"]["train_params"]["encryption"]["plain"]
        else:
            sec_conf = assist_conf["train_info"]["train_params"]["encryption"]["otp"]

        if encryption_method == "plain":
            fed_method = AggregationPlainLeaf(sec_conf)
            fed_assist_method = AggregationPlainRoot(sec_conf)

        elif encryption_method == "otp":
            mocker.patch.object(DualChannel, "__init__", return_value=None)
            DualChannel.remote_id = "node-2"
            supported_shortest_exponents = [225, 275, 325, 375, 400]
            shorest_exponent = supported_shortest_exponents[1]
            lower_bound = 1 << (supported_shortest_exponents[1] - 1)
            upper_bound = 1 << shorest_exponent
            primes = [int(p.replace(' ', ''), 16) for p in primes_hex]
            rand_num_generator = SystemRandom()
            a = rand_num_generator.randint(lower_bound, upper_bound)
            g_power_a = powmod(2, a, primes[1])
            mocker.patch.object(DualChannel, "swap",
                                return_value=(1, g_power_a))
            mocker.patch.object(Commu, "node_id", "node-1")
            fed_method = AggregationOTPLeaf(sec_conf)
            fed_assist_method = AggregationOTPRoot(sec_conf)

        print(f"trainer conf: {json.dumps(conf)}")
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
        prt = HorizontalPoissonRegressionLabelTrainer(conf)
        prt.model.linear.weight = torch.nn.parameter.Parameter(
            torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]))
        prt.model.linear.bias = torch.nn.parameter.Parameter(
            torch.tensor([0.0]))
        prt_a = HorizontalPoissonRegressionAssistTrainer(assist_conf)
        prt_a.model.linear.weight = torch.nn.parameter.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
        print(prt_a.model.linear.weight)
        prt_a.model.linear.bias = torch.nn.parameter.Parameter(
            torch.tensor([0.0]))
        esflag_recv = pickle.dumps(False) + EOV
        params_plain_recv = pickle.dumps(prt_a.model.state_dict()) + EOV
        print("param plain received")
        print(params_plain_recv)
        print("param plain received loaded")
        print(pickle.loads(params_plain_recv))
        params_send = fed_method._calc_upload_value(
            prt.model.state_dict(), len(prt.train_dataloader.dataset))
        params_collect = pickle.dumps(params_send)
        print(f"Params collect: {params_collect}")
        print(f"Loaded params: {pickle.loads(params_collect)}")
        print()
        # agg_otp = fed_assist_method._calc_aggregated_params(
        #     list(map(lambda x: pickle.loads(x), [params_collect, params_collect])))
        # print(f"agg otp: {agg_otp}")
        agg_otp = prt_a.model.state_dict()

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
            AggregationOTPRoot, "aggregate", side_effect=mock_agg
        )
        mocker.patch.object(
            AggregationPlainRoot, "aggregate", side_effect=mock_agg
        )
        mocker.patch("service.fed_control._send_progress")

        print(prt.model)
        prt.model.linear.weight = torch.nn.parameter.Parameter(
            torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]))
        prt.model.linear.bias = torch.nn.parameter.Parameter(
            torch.tensor([0.0]))
        prt.fit()
        print("Successfully tested label trainer")
        prt_a.model.linear.weight = torch.nn.parameter.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
        print(prt_a.model.linear.weight)
        prt_a.model.linear.bias = torch.nn.parameter.Parameter(
            torch.tensor([0.0]))
        print(prt_a.model.linear.bias)
        prt_a.fit()
