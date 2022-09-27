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
from scipy.stats import normaltest
import pandas as pd
import torch
import pytest

import service.fed_config
from algorithm.core.horizontal.aggregation.aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf

from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.key_agreement.contants import primes_hex
from gmpy2 import powmod

from common.utils.logger import logger


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
        conf = json.load(f)[0]
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_data.csv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_nbafl/trainer.json") as f:
        conf = json.load(f)[0]
        print(conf["input"]["trainset"])
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.csv"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
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

    def test_uplink_sigma(self, get_trainer_conf, mocker):
        conf = get_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.nbafl.label_trainer import HorizontalNbaflLabelTrainer
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            AggregationOTPLeaf, "__init__", return_value=None
        )
        nbafl_t = HorizontalNbaflLabelTrainer(conf)
        logger.info(f"{len(nbafl_t.train_dataloader.dataset)} of data")
        nbafl_t._calc_uplink_sigma({})
        sigma_u = nbafl_t.sigma_u
        expected_sigma_u = np.sqrt(2 * np.log(12.5)) / 80
        logger.info(f"expected uplink sigma: {expected_sigma_u}")
        assert np.abs(sigma_u - expected_sigma_u) < 0.0001

    def test_uplink_add_noise(self, get_trainer_conf, mocker):
        conf = get_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.nbafl.label_trainer import HorizontalNbaflLabelTrainer
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            AggregationOTPLeaf, "__init__", return_value=None
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

    def test_downlink_sigma(self, get_assist_trainer_conf, mocker):
        conf = get_assist_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.nbafl.assist_trainer import HorizontalNbaflAssistTrainer
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            AggregationOTPRoot, "__init__", return_value=None
        )
        nbafl_at = HorizontalNbaflAssistTrainer(conf)
        nbafl_at.min_sample_num = 10
        expected_sigma_d = 10 * \
            np.sqrt(2 * np.log(12.5)) * np.sqrt((25-8) / 20)
        nbafl_at._calc_downlink_sigma({})
        assert (nbafl_at.sigma_d - expected_sigma_d) < 0.0001

    def test_label_trainer(self, get_trainer_conf, mocker):
        conf = get_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.nbafl.label_trainer import HorizontalNbaflLabelTrainer
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            AggregationOTPLeaf, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        mocker.patch.object(
            HorizontalNbaflLabelTrainer, "_download_model", return_value=None
        )
        mocker.patch.object(
            AggregationOTPLeaf, "upload", return_value=None
        )
        nbafl_t = HorizontalNbaflLabelTrainer(conf)
        nbafl_t.sigma_u = 0.1
        nbafl_t.train_loop()

    def test_assist_trainer(self, get_assist_trainer_conf, mocker):
        conf = get_assist_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.nbafl.assist_trainer import HorizontalNbaflAssistTrainer
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "recv", return_value=10
        )
        mocker.patch.object(
            AggregationOTPRoot, "__init__", return_value=None
        )
        mocker.patch.object(
            AggregationOTPRoot, "broadcast", return_value=None
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        nbafl_at = HorizontalNbaflAssistTrainer(conf)
        model_state_dict = nbafl_at.model.state_dict()
        mocker.patch.object(
            AggregationOTPRoot, "aggregate", return_value=model_state_dict
        )
        nbafl_at.min_sample_num = 10
        nbafl_at.train_loop()
