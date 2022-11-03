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


import copy
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import tenseal as ts
import torch

import service.fed_config
from algorithm.framework.vertical.logistic_regression.label_trainer import \
    VerticalLogisticRegressionLabelTrainer
from algorithm.framework.vertical.logistic_regression.trainer import \
    VerticalLogisticRegressionTrainer
from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier
from common.communication.gRPC.python.commu import Commu


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
    case_df[['y', 'x0', 'x1', 'x2']].head(80).to_csv(
        "/opt/dataset/unit_test/train_guest.csv", index=True
    )
    case_df[['y', 'x0', 'x1', 'x2']].tail(20).to_csv(
        "/opt/dataset/unit_test/test_guest.csv", index=True
    )
    case_df[['x3', 'x4']].head(80).to_csv(
        "/opt/dataset/unit_test/train_host.csv", index=True
    )
    case_df[['x3', 'x4']].tail(20).to_csv(
        "/opt/dataset/unit_test/test_host.csv", index=True
    )


config_sync = {
    "train_info": {
        "interaction_params": {
            "save_frequency": -1,
            "write_training_prediction": True,
            "write_validation_prediction": True,
            "echo_training_metrics": True
        },
        "train_params": {
            "global_epoch": 10,
            "batch_size": 2048,
            "encryption": {
                "ckks": {
                    "poly_modulus_degree": 8192,
                    "coeff_mod_bit_sizes": [
                        60,
                        40,
                        40,
                        60
                    ],
                    "global_scale_bit_size": 40
                }
            },
            "optimizer": {
                "lr": 0.01,
                "p": 2,
                "alpha": 1e-4
            },
            "early_stopping": {
                "key": "acc",
                "patience": 10,
                "delta": 0
            },
            "random_seed": None
        }
    }
}


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/vertical_logistic_regression/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_guest.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_guest.csv"
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
        conf["train_info"]["interaction_params"]["save_frequency"] = -1
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/vertical_logistic_regression/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_host.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_host.csv"
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
        # conf["train_info"]["interaction_params"]["save_frequency"] = -1
    yield conf


@pytest.fixture(scope="module", autouse=True)
def env():
    Commu.node_id = "node-1"
    Commu.trainer_ids = ['node-1', 'node-2']
    Commu.scheduler_id = 'assist_trainer'
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    # if not os.path.exists("/opt/config/unit_test"):
    # 	os.makedirs("/opt/config/unit_test")
    prepare_data()
    yield
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    # if os.path.exists("/opt/config/unit_test"):
    # 	shutil.rmtree("/opt/config/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")


class TestLogisticRegression:
    @pytest.mark.parametrize("encryption_method, p", [
        ("ckks", 1), ("paillier", 1), ("plain", 1), ("other", 1),
        ("ckks", 0), ("paillier", 0), ("plain", 0),
        ("ckks", 2), ("paillier", 2), ("plain", 2), ("ckks", 3)
    ])
    def test_label_trainer(self, get_label_trainer_conf, p, encryption_method, mocker):
        # label trainer 流程测试
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )

        lrt = VerticalLogisticRegressionLabelTrainer(get_label_trainer_conf)

        mocker.patch.object(
            BroadcastChannel, "scatter", return_value=0
        )
        if encryption_method == "paillier":
            lrt.encryption_config = {
                "paillier": {
                    "key_bit_size": 2048,
                    "precision": 7,
                    "djn_on": True,
                    "parallelize_on": True
                }
            }
        elif encryption_method == "plain":
            lrt.encryption_config = {
                "plain": {}
            }
        elif encryption_method == "ckks":
            pass
        else:
            lrt.encryption_config = {
                encryption_method: {}
            }

        encryption_config = lrt.encryption_config
        if encryption_method == "ckks":
            private_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=encryption_config[encryption_method]["poly_modulus_degree"],
                coeff_mod_bit_sizes=encryption_config[encryption_method]["coeff_mod_bit_sizes"]
            )
            private_context.generate_galois_keys()
            private_context.generate_relin_keys()
            private_context.global_scale = 1 << encryption_config[
                encryption_method]["global_scale_bit_size"]
            serialized_public_context = private_context.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True
            )
            public_context = ts.context_from(serialized_public_context)
        elif encryption_method == "paillier":
            private_context = Paillier.context(
                encryption_config[encryption_method]["key_bit_size"], djn_on=encryption_config[encryption_method]["djn_on"])
            public_context = private_context.to_public().serialize()
            public_context = Paillier.context_from(public_context)

        def mock_collect(*args, **kwargs):
            if mock_channel_collect.call_count <= 1:
                return [2]
            if encryption_method == "ckks":
                if mock_channel_collect.call_count > 10:
                    return []
                if mock_channel_collect.call_count % 3 == 2:
                    return [torch.tensor(np.zeros([80, 1]))]
                elif mock_channel_collect.call_count % 3 == 0:
                    pred_residual = torch.tensor(np.random.random(2))
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    pred_residual = ts.ckks_vector_from(
                        public_context, serialized_enc_pred_residual)
                    return [pred_residual.serialize()]
                else:
                    return [torch.tensor(np.zeros([20, 1]))]
            elif encryption_method == "paillier":
                return []
            elif encryption_method == "plain":
                if mock_channel_collect.call_count >= 10:
                    return []
                if mock_channel_collect.call_count % 2 == 0:
                    return [torch.tensor(np.zeros([80, 1]))]
                else:
                    return [torch.tensor(np.zeros([20, 1]))]
            else:
                pass

        mock_channel_collect = mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_collect
        )

        lrt.optimizer_config['p'] = p
        if encryption_method not in ("ckks", "paillier", "plain"):
            msg = f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'ckks', 'plain'."
            with pytest.raises(ValueError) as e:
                lrt.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == msg
        elif p not in (0, 1, 2):
            with pytest.raises(NotImplementedError) as e:
                lrt.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == "Regular P={} not implement.".format(p)
        else:
            lrt.fit()
        self.check_model_output()

    @pytest.mark.parametrize("encryption_method, p", [
        ("ckks", 1), ("paillier", 1), ("plain", 1), ("other", 1),
        ("ckks", 0), ("paillier", 0), ("plain", 0),
        ("ckks", 2), ("paillier", 2), ("plain", 2), ("ckks", 3)
    ])
    def test_trainer(self, get_trainer_conf, encryption_method, p, mocker):
        # trainer 流程测试
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        def mock_func(*args, **kwargs):
            if mock_broadcast_recv.call_count == 1:
                return copy.deepcopy(config_sync)
            elif mock_broadcast_recv.call_count == 2:
                return 50
            else:
                return 0

        mock_broadcast_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_func
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        lrt = VerticalLogisticRegressionTrainer(get_trainer_conf)

        if encryption_method == "paillier":
            lrt.encryption_config = {
                "paillier": {
                    "key_bit_size": 2048,
                    "precision": 7,
                    "djn_on": True,
                    "parallelize_on": True
                }
            }
        elif encryption_method == "plain":
            lrt.encryption_config = {
                "plain": {}
            }
        elif encryption_method == "ckks":
            pass
        else:
            lrt.encryption_config = {
                encryption_method: {}
            }

        encryption_config = lrt.encryption_config
        if encryption_method == "ckks":
            private_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=encryption_config[encryption_method]["poly_modulus_degree"],
                coeff_mod_bit_sizes=encryption_config[encryption_method]["coeff_mod_bit_sizes"]
            )
            private_context.generate_galois_keys()
            private_context.generate_relin_keys()
            private_context.global_scale = 1 << encryption_config[
                encryption_method]["global_scale_bit_size"]
        elif encryption_method == "paillier":
            num_cores = - \
                1 if encryption_config[encryption_method]["parallelize_on"] else 1
            private_context = Paillier.context(
                encryption_config[encryption_method]["key_bit_size"], djn_on=encryption_config[encryption_method]["djn_on"])

        def mock_predict_residual(*args, **kwargs):
            if encryption_method == "ckks":
                if mock_channel_recv.call_count <= 1:
                    serialized_public_context = private_context.serialize(
                        save_public_key=True,
                        save_secret_key=False,
                        save_galois_keys=True,
                        save_relin_keys=True
                    )
                    return serialized_public_context
                elif mock_channel_recv.call_count % 3 == 2:
                    pred_residual = torch.tensor(np.random.random(80))
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    return serialized_enc_pred_residual
                elif mock_channel_recv.call_count % 3 == 0:
                    return np.random.random(2)
                else:
                    return False, False, -1
            elif encryption_method == "paillier":
                if mock_channel_recv.call_count <= 1:
                    return private_context.to_public().serialize()
                elif mock_channel_recv.call_count % 3 == 2:
                    pred_residual = torch.tensor(np.random.random(80))
                    enc_pred_residual = Paillier.encrypt(
                        private_context,
                        pred_residual.numpy().astype(np.float32).flatten(),
                        precision=encryption_config["paillier"]["precision"],
                        obfuscation=True,
                        num_cores=num_cores
                    )
                    return Paillier.serialize(enc_pred_residual)
                elif mock_channel_recv.call_count % 3 == 0:
                    return np.random.random(2)
                else:
                    return False, False, -1
            elif encryption_method == "plain":
                if mock_channel_recv.call_count % 2 == 1:
                    return torch.tensor(np.random.random((80, 1)), dtype=torch.float)
                else:
                    return False, False, -1

        mock_channel_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_predict_residual
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )
        lrt.optimizer_config['p'] = p

        if encryption_method not in ("ckks", "paillier", "plain"):
            msg = f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'ckks', 'plain'."
            with pytest.raises(ValueError) as e:
                lrt.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == msg
        elif p not in (0, 1, 2):
            with pytest.raises(NotImplementedError) as e:
                lrt.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == "Regular P={} not implement.".format(p)
        else:
            lrt.fit()
        self.check_model_output()

    @pytest.mark.parametrize("encryption_method", ["ckks"])
    def test_early_stopping(self, get_label_trainer_conf, get_trainer_conf, encryption_method, mocker):
        # 早停测试
        get_label_trainer_conf["train_info"]["train_params"]["early_stopping"]["patience"] = 1
        get_label_trainer_conf["train_info"]["train_params"]["early_stopping"]["delta"] = 1e-3
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )
        mocker.patch.object(
            BroadcastChannel, "scatter", return_value=0
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        lrt = VerticalLogisticRegressionLabelTrainer(get_label_trainer_conf)
        encryption_config = lrt.encryption_config

        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=encryption_config["ckks"]["poly_modulus_degree"],
            coeff_mod_bit_sizes=encryption_config["ckks"]["coeff_mod_bit_sizes"]
        )
        private_context.generate_galois_keys()
        private_context.generate_relin_keys()
        private_context.global_scale = 1 << encryption_config["ckks"]["global_scale_bit_size"]
        serialized_public_context = private_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True
        )
        public_context = ts.context_from(serialized_public_context)

        def mock_collect(*args, **kwargs):
            if encryption_method == "ckks":
                print(mock_channel_collect.call_count)
                if mock_channel_collect.call_count >= 8:
                    return []
                if mock_channel_collect.call_count % 3 == 1:
                    return [torch.tensor(np.zeros([80, 1]))]
                elif mock_channel_collect.call_count % 3 == 2:
                    pred_residual = torch.tensor(np.random.random(2))
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    pred_residual = ts.ckks_vector_from(
                        public_context, serialized_enc_pred_residual)
                    return [pred_residual.serialize()]
                else:
                    return [torch.tensor(np.zeros([20, 1]))]
            elif encryption_method == "paillier":
                return []

        mock_channel_collect = mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_collect
        )
        mocker.patch.object(
            lrt, "check_data", return_value=None
        )
        lrt.fit()

        def mock_func(*args, **kwargs):
            if mock_broadcast_recv.call_count == 1:
                return copy.deepcopy(config_sync)
            elif mock_broadcast_recv.call_count == 2:
                return 50
            else:
                return 0

        mock_broadcast_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_func
        )

        trainer = VerticalLogisticRegressionTrainer(get_trainer_conf)

        encryption_config = trainer.encryption_config
        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=encryption_config["ckks"]["poly_modulus_degree"],
            coeff_mod_bit_sizes=encryption_config["ckks"]["coeff_mod_bit_sizes"]
        )
        private_context.generate_galois_keys()
        private_context.generate_relin_keys()
        private_context.global_scale = 1 << encryption_config["ckks"]["global_scale_bit_size"]

        def mock_predict_residual(*args, **kwargs):
            if mock_channel_recv.call_count <= 1:
                serialized_public_context = private_context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=True,
                    save_relin_keys=True
                )
                return serialized_public_context
            elif mock_channel_recv.call_count % 3 == 2:
                pred_residual = torch.tensor(np.random.random(80))
                enc_pred_residual = ts.ckks_vector(
                    private_context, pred_residual.numpy().flatten())
                serialized_enc_pred_residual = enc_pred_residual.serialize()
                return serialized_enc_pred_residual
            elif mock_channel_recv.call_count % 3 == 0:
                return np.random.random(2)
            else:
                return True, True, 1

        mock_channel_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_predict_residual
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )

        trainer.fit()

    @pytest.mark.parametrize("encryption_method", ["ckks"])
    def test_save_frequency(self, get_label_trainer_conf, get_trainer_conf, encryption_method, mocker):
        # 测试模型留存频率参数是否生效
        get_label_trainer_conf["train_info"]["interaction_params"]["save_frequency"] = 1
        get_trainer_conf["train_info"]["interaction_params"] = {}
        get_trainer_conf["train_info"]["interaction_params"]["save_frequency"] = 1
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        def mock_collect(*args, **kwargs):
            if encryption_method == "ckks":
                if mock_channel_collect.call_count > 9:
                    return []
                if mock_channel_collect.call_count % 3 == 1:
                    return [torch.tensor(np.zeros([80, 1]))]
                elif mock_channel_collect.call_count % 3 == 2:
                    pred_residual = torch.tensor(np.random.random(2))
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    pred_residual = ts.ckks_vector_from(
                        public_context, serialized_enc_pred_residual)
                    return [pred_residual.serialize()]
                else:
                    return [torch.tensor(np.zeros([20, 1]))]
            elif encryption_method == "paillier":
                return []

        mock_channel_collect = mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_collect
        )
        mocker.patch.object(
            BroadcastChannel, "scatter", return_value=0
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        lrt = VerticalLogisticRegressionLabelTrainer(get_label_trainer_conf)
        encryption_config = lrt.encryption_config

        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=encryption_config["ckks"]["poly_modulus_degree"],
            coeff_mod_bit_sizes=encryption_config["ckks"]["coeff_mod_bit_sizes"]
        )
        private_context.generate_galois_keys()
        private_context.generate_relin_keys()
        private_context.global_scale = 1 << encryption_config["ckks"]["global_scale_bit_size"]
        serialized_public_context = private_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True
        )
        public_context = ts.context_from(serialized_public_context)
        mocker.patch.object(
            lrt, "check_data", return_value=None
        )
        lrt.fit()

        def mock_func(*args, **kwargs):
            if mock_broadcast_recv.call_count == 1:
                return config_sync
            elif mock_broadcast_recv.call_count == 2:
                return 50
            else:
                return 0

        mock_broadcast_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_func
        )
        trainer = VerticalLogisticRegressionTrainer(get_trainer_conf)
        encryption_config = trainer.encryption_config
        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=encryption_config["ckks"]["poly_modulus_degree"],
            coeff_mod_bit_sizes=encryption_config["ckks"]["coeff_mod_bit_sizes"]
        )
        private_context.generate_galois_keys()
        private_context.generate_relin_keys()
        private_context.global_scale = 1 << encryption_config["ckks"]["global_scale_bit_size"]

        def mock_predict_residual(*args, **kwargs):
            if mock_channel_recv.call_count <= 1:
                serialized_public_context = private_context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=True,
                    save_relin_keys=True
                )
                return serialized_public_context
            elif mock_channel_recv.call_count % 3 == 2:
                pred_residual = torch.tensor(np.random.random(80))
                enc_pred_residual = ts.ckks_vector(
                    private_context, pred_residual.numpy().flatten())
                serialized_enc_pred_residual = enc_pred_residual.serialize()
                return serialized_enc_pred_residual
            elif mock_channel_recv.call_count % 3 == 0:
                return np.random.random(2)
            else:
                return False, False, -1

        mock_channel_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_predict_residual
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )
        trainer.fit()

    @pytest.mark.parametrize("encryption_method", ["ckks"])
    def test_save_path(self, get_label_trainer_conf, encryption_method, mocker):
        # 假如留存目录不存在，是否会自动创建完成运行\
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )
        mocker.patch.object(
            BroadcastChannel, "scatter", return_value=0
        )
        get_label_trainer_conf["output"]["path"] = "/opt/checkpoints/unit_test_2"

        def mock_collect(*args, **kwargs):
            if encryption_method == "ckks":
                if mock_channel_collect.call_count > 9:
                    return []
                if mock_channel_collect.call_count % 3 == 1:
                    return [torch.tensor(np.zeros([80, 1]))]
                elif mock_channel_collect.call_count % 3 == 2:
                    pred_residual = torch.tensor(np.random.random(2))
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    pred_residual = ts.ckks_vector_from(
                        public_context, serialized_enc_pred_residual)
                    return [pred_residual.serialize()]
                else:
                    return [torch.tensor(np.zeros([20, 1]))]
            elif encryption_method == "paillier":
                return []

        mock_channel_collect = mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_collect
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        lrt = VerticalLogisticRegressionLabelTrainer(get_label_trainer_conf)
        encryption_config = lrt.encryption_config
        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=encryption_config["ckks"]["poly_modulus_degree"],
            coeff_mod_bit_sizes=encryption_config["ckks"]["coeff_mod_bit_sizes"]
        )
        private_context.generate_galois_keys()
        private_context.generate_relin_keys()
        private_context.global_scale = 1 << encryption_config["ckks"]["global_scale_bit_size"]
        serialized_public_context = private_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True
        )
        public_context = ts.context_from(serialized_public_context)
        mocker.patch.object(
            lrt, "check_data", return_value=None
        )
        lrt.fit()
        shutil.rmtree("/opt/checkpoints/unit_test_2")

    @staticmethod
    def check_model_output():
        # 检查是否正常输出了model_config.json
        assert os.path.exists("/opt/checkpoints/unit_test/model_config.json")
        with open("/opt/checkpoints/unit_test/model_config.json") as f:
            model_config = json.load(f)

        # 检查model_config.json的stage是否符合预期
        assert model_config[-1]["class_name"] == "VerticalLogisticRegression"

        filename = "/opt/checkpoints/unit_test/" + model_config[0]["filename"]
        dim = model_config[-1]["input_dim"]
        bias = model_config[-1]["bias"]

        if bias:
            assert dim == 3
        else:
            assert dim == 2

        # 检查是否写出了模型文件，模型文件是否合法
        assert os.path.exists(filename)

        model = torch.load(filename)

        assert model["state_dict"]["linear.weight"].shape[1] == dim

        if bias:
            assert "linear.bias" in model["state_dict"]
        else:
            assert "linear.bias" not in model["state_dict"]
