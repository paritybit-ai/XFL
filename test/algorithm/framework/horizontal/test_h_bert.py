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
import pandas as pd
import pickle
from collections import OrderedDict

import numpy as np
import pytest

import service.fed_config
from algorithm.core.horizontal.aggregation.aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from algorithm.framework.horizontal.bert.assist_trainer import HorizontalBertAssistTrainer
from algorithm.framework.horizontal.bert.label_trainer import HorizontalBertLabelTrainer
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.key_agreement.contants import primes_hex
from gmpy2 import powmod

MOV = b"@" # middle of value
EOV = b"&" # end of value

def prepare_data():
    case_df = pd.DataFrame({
        'sentence': ["the action is stilted","cold movie","smile on face","redundant concept","the greatest musicians",
                    "sometimes dry","shot on ugly digital video","funny yet","a beautifully","no apparent joy"],
        'label': [0,1,0,1,1,0,1,1,1,1]
    })
    case_df.head(8).to_csv(
        "/opt/dataset/unit_test/train_data.tsv", sep='\t'
    )
    case_df.tail(2).to_csv(
        "/opt/dataset/unit_test/test_data.tsv", sep='\t'
    )


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_bert/assist_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_data.tsv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
        conf["model_info"]["config"]["from_pretrained"] = False
        conf["model_info"]["config"]["hidden_size"] = 144
        conf["model_info"]["config"]["num_hidden_layers"] = 12
        conf["model_info"]["config"]["num_attention_head"] = 12
        conf["model_info"]["config"]["intermediate_size"] = 144
        conf["train_info"]["params"]["batch_size"] = 2
        conf["train_info"]["params"]["global_epoch"] = 2

    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_bert/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.tsv"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
        conf["model_info"]["config"]["from_pretrained"] = False
        conf["model_info"]["config"]["hidden_size"] = 144
        conf["model_info"]["config"]["num_hidden_layers"] = 12
        conf["model_info"]["config"]["num_attention_head"] = 12
        conf["model_info"]["config"]["intermediate_size"] = 144
        conf["train_info"]["params"]["batch_size"] = 2
        conf["train_info"]["params"]["global_epoch"] = 2
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


class TestBert:
    #@pytest.mark.skip(reason="no reason")
    @pytest.mark.parametrize("encryption_method", ['plain']) # ['otp', 'plain'] otp too slow
    def test_trainer(self, get_trainer_conf, get_assist_trainer_conf, encryption_method, mocker):
        fed_method = None
        fed_assist_method = None
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=['node-1', 'node-2']
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_assist_trainer", return_value='assist_trainer'
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "node_id", 'node-1'
        )
        if encryption_method == "plain":
            conf["train_info"]["params"]["aggregation_config"]["encryption"] = {
                "method": "plain"}
            assist_conf["train_info"]["params"]["aggregation_config"]["encryption"] = {
                "method": "plain"}

        sec_conf = conf["train_info"]["params"]["aggregation_config"]["encryption"]

        def mock_recv(*args, **kwargs):
            return params_plain_recv

        def mock_collect(*args, **kwargs):
            return params_collect

        def mock_agg(*args, **kwargs):
            return agg_otp
        
        if encryption_method == "plain":
            fed_method = AggregationPlainLeaf(sec_conf)
            fed_assist_method = AggregationPlainRoot(sec_conf)
        elif encryption_method == "otp":
            mocker.patch.object(DualChannel, "__init__", return_value=None)
            # dc = DualChannel(name="otp_diffie_hellman", ids=['node-1', 'node-2'])
            DualChannel.remote_id = "node-2"
            supported_shortest_exponents = [225, 275, 325, 375, 400]
            shorest_exponent = supported_shortest_exponents[1]
            lower_bound = 1 << (supported_shortest_exponents[1] - 1)
            upper_bound = 1 << shorest_exponent
            primes = [int(p.replace(' ', ''), 16) for p in primes_hex]
            rand_num_generator = SystemRandom()
            a = rand_num_generator.randint(lower_bound, upper_bound)
            g_power_a = powmod(2, a, primes[1])
            mocker.patch.object(DualChannel, "swap", return_value=(1, g_power_a))
            fed_method = AggregationOTPLeaf(sec_conf)
            fed_assist_method = AggregationOTPRoot(sec_conf)

        bert = HorizontalBertLabelTrainer(conf)
        bert_a = HorizontalBertAssistTrainer(assist_conf)
        
        params_plain_recv = pickle.dumps(OrderedDict({i:w for i,w in enumerate(bert_a.model.get_weights())})) + EOV
        params_send = fed_method._calc_upload_value(
            OrderedDict({i:w for i,w in enumerate(bert.model.get_weights())}), len(bert.train_dataloader._input_dataset))
        params_collect = pickle.dumps(params_send)
        agg_otp = fed_assist_method._calc_aggregated_params(list(map(lambda x: pickle.loads(x), [params_collect,params_collect])))
        del params_send, params_collect

        def mock_recv(*args, **kwargs):
            if recv_mocker.call_count % 4 in [1,2]:
                return params_plain_recv
            elif recv_mocker.call_count % 4 in [0,3] :
                return params_collect

        recv_mocker = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_recv
        )
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )
        mocker.patch.object(
            AggregationOTPRoot, "aggregate", side_effect=mock_agg
        )
        mocker.patch.object(
            AggregationPlainRoot, "aggregate", side_effect=mock_agg
        )

        bert.fit()
        bert_a.fit()