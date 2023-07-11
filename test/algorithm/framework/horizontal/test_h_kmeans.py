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
import pandas as pd
import pytest
from gmpy2 import powmod
from sklearn.datasets import make_blobs

from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.core.horizontal.aggregation import aggregation_base
from algorithm.core.horizontal.aggregation.aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from algorithm.framework.horizontal.kmeans.assist_trainer import HorizontalKmeansAssistTrainer
from algorithm.framework.horizontal.kmeans.label_trainer import HorizontalKmeansLabelTrainer
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.key_agreement.contants import primes_hex


MOV = b"@"  # middle of value
EOV = b"&"  # end of value


def prepare_data():
    X, y = make_blobs(n_samples=450, n_features=2,
                      random_state=42, cluster_std=2.0)
    data_df = pd.DataFrame({'label': y, 'x1': X[:, 0], 'x2': X[:, 1]})
    data_df.head(400).to_csv(
        "/opt/dataset/unit_test/horizontal_kmeans_train.csv"
    )
    data_df.tail(50).to_csv(
        "/opt/dataset/unit_test/horizontal_kmeans_test.csv"
    )


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_kmeans/assist_trainer.json") as f:
        conf = json.load(f)

    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_kmeans/trainer.json") as f:
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


class TestHorizontalKMeans:
    # ['otp', 'plain'] otp too slow
    @pytest.mark.parametrize("encryption_method", ['plain'])
    def test_trainer(self, get_trainer_conf, get_assist_trainer_conf, encryption_method, mocker):
        mocker.patch.object(aggregation_base, "MAX_BLOCK_SIZE", 30000000)
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
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        
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

        kmeans = HorizontalKmeansLabelTrainer(conf)
        kmeans_a = HorizontalKmeansAssistTrainer(assist_conf)
        esflag_recv = pickle.dumps(False) + EOV
        params_plain_recv = pickle.dumps(kmeans_a.model.state_dict()) + EOV
        params_send = fed_method._calc_upload_value(
            kmeans.model.state_dict(), len(kmeans.train_dataloader.dataset))
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
            AggregationOTPRoot, "aggregate", side_effect=mock_agg
        )
        mocker.patch.object(
            AggregationPlainRoot, "aggregate", side_effect=mock_agg
        )
        mocker.patch("service.fed_control._send_progress")
        kmeans.fit()
        kmeans_a.fit()
