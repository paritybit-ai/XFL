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
import pytest

import service.fed_config
from algorithm.core.horizontal.aggregation.aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.key_agreement.contants import primes_hex
from gmpy2 import powmod

from dgllife.data import ESOL, ToxCast, BACE, HIV
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer

from dgllife.utils import ScaffoldSplitter, RandomSplitter
from dgllife.model import GCNPredictor
from dgllife.utils import EarlyStopping, Meter, SMILESToBigraph

from sklearn.model_selection import train_test_split

MOV = b"@"  # middle of value
EOV = b"&"  # end of value


def prepare_data():
    node_featurizer = CanonicalAtomFeaturizer()
    edge_featurizer = None  # CanonicalBondFeaturizer()

    smiles_to_g = SMILESToBigraph(
        add_self_loop=True,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer
    )
    data = HIV(smiles_to_graph=smiles_to_g, n_jobs=1,
               cache_file_path="/opt/dataset/unit_test/dgl_hiv.bin")

    df = data.df
    df = df[['HIV_active', 'smiles']]
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df.to_csv("/opt/dataset/unit_test/train_data.csv", index=False)
    test_df.to_csv("/opt/dataset/unit_test/test_data.csv", index=False)


@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_gcn_mol/assist_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_data.csv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
        conf["model_info"]["config"]["layers"] = "unit_test"
        conf["train_info"]["params"]["batch_size"] = 16
        conf["train_info"]["params"]["global_epoch"] = 2
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_gcn_mol/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.csv"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
        conf["model_info"]["config"]["layers"] = "unit_test"
        conf["train_info"]["params"]["batch_size"] = 16
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


class TestGcnMol:
    # ['otp', 'plain'] otp too slow
    @pytest.mark.parametrize("encryption_method", ['plain'])
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
            mocker.patch.object(DualChannel, "swap",
                                return_value=(1, g_power_a))
            fed_method = AggregationOTPLeaf(sec_conf)
            fed_assist_method = AggregationOTPRoot(sec_conf)

        service.fed_config.FedConfig.stage_config = conf
        from algorithm.framework.horizontal.gcn_mol.assist_trainer import HorizontalGcnMolAssistTrainer
        from algorithm.framework.horizontal.gcn_mol.label_trainer import HorizontalGcnMolLabelTrainer
        rest = HorizontalGcnMolLabelTrainer(conf)
        rest_a = HorizontalGcnMolAssistTrainer(assist_conf)
        params_plain_recv = pickle.dumps(rest_a.model.state_dict()) + EOV
        params_send = fed_method._calc_upload_value(
            rest.model.state_dict(), len(rest.train_dataloader.dataset))
        params_collect = pickle.dumps(params_send)
        agg_otp = fed_assist_method._calc_aggregated_params(
            list(map(lambda x: pickle.loads(x), [params_collect, params_collect])))

        def mock_recv(*args, **kwargs):
            if recv_mocker.call_count % 4 in [1, 2]:
                return params_plain_recv
            elif recv_mocker.call_count % 4 in [0, 3]:
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
        mocker.patch("service.fed_control._send_progress")
        rest.fit()
        rest_a.fit()