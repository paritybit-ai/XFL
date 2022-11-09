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
import tenseal as ts
import numpy as np
import pandas as pd
import pytest
from pytest_mock import mocker

from common.crypto.paillier.paillier import Paillier
from service.fed_config import FedConfig
import service.fed_config
from algorithm.framework.vertical.linear_regression.assist_trainer import \
    VerticalLinearRegressionAssistTrainer
from algorithm.framework.vertical.linear_regression.trainer import VerticalLinearRegressionTrainer
from algorithm.framework.vertical.linear_regression.label_trainer import VerticalLinearRegressionLabelTrainer
from common.communication.gRPC.python.channel import (BroadcastChannel, DualChannel)


def prepare_data():
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': 2 * np.random.random(1000) + 2.0,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = case_df['x0'] + case_df['x1'] + case_df['x3'] + 0.5 * case_df['x2']
    case_df[['y', 'x0']].head(800).to_csv(
        "/opt/dataset/unit_test/train_guest.csv", index=True
    )
    case_df[['y', 'x0']].tail(200).to_csv(
        "/opt/dataset/unit_test/test_guest.csv", index=True
    )
    case_df[['x1', 'x2']].head(800).to_csv(
        "/opt/dataset/unit_test/train_host1.csv", index=True
    )
    case_df[['x1', 'x2']].tail(200).to_csv(
        "/opt/dataset/unit_test/test_host1.csv", index=True
    )
    case_df[['x3', 'x4']].head(800).to_csv(
        "/opt/dataset/unit_test/train_host2.csv", index=True
    )
    case_df[['x3', 'x4']].tail(200).to_csv(
        "/opt/dataset/unit_test/test_host2.csv", index=True
    )


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


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/vertical_linear_regression/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_guest.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_guest.csv"
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


@pytest.fixture()
def get_trainer1_conf():
    with open("python/algorithm/config/vertical_linear_regression/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_host1.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_host1.csv"
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


class TestVerticalLinearRegressionTrainer:
    @pytest.mark.parametrize("encryption", [{"ckks": {
        "poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60], "global_scale_bit_size": 40}},
        {"plain": {}}, {"paillier": {"key_bit_size": 2048, "precision": 7, "djn_on": True, "parallelize_on": True}}])
    def test_all_trainers(self, get_label_trainer_conf, encryption, mocker):
        conf = get_label_trainer_conf
        with open("python/algorithm/config/vertical_linear_regression/trainer.json") as f:
            conf_t = json.load(f)
            conf_t["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
            conf_t["input"]["trainset"][0]["name"] = "train_host1.csv"
            conf_t["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
            conf_t["input"]["valset"][0]["name"] = "test_host1.csv"
            conf_t["output"]["path"] = "/opt/checkpoints/unit_test"

        conf["train_info"]["train_params"]["global_epoch"] = 1
        conf["train_info"]["train_params"]["batch_size"] = 1000
        conf["train_info"]["train_params"]["encryption"] = encryption
        conf_t["train_info"]["train_params"]["batch_size"] = 1000
        conf_t["train_info"]["train_params"]["encryption"] = encryption
        conf_t["train_info"]["train_params"]["global_epoch"] = 1

        # test trainset not configured error
        conf2 = copy.deepcopy(conf)
        conf2["input"]["trainset"] = []
        with pytest.raises(NotImplementedError) as e:
            vlr_ = VerticalLinearRegressionLabelTrainer(conf2)
            exec_msg = e.value.args[0]
            assert exec_msg == "Trainset was not configured."

        # test trainset type not configured error
        conf1 = copy.deepcopy(conf)
        conf1["input"]["trainset"][0]["type"] = "json"
        with pytest.raises(NotImplementedError) as e:
            vlr_ = VerticalLinearRegressionLabelTrainer(conf1)
            exec_msg = e.value.args[0]
            assert exec_msg == "Dataset type {} is not supported.".format(vlr_.input["trainset"][0]["type"])

        # mock label_trainer
        encryption_config = encryption
        encryption_method = list(encryption.keys())[0]

        if encryption_method == "ckks":
            private_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=encryption_config[encryption_method]["poly_modulus_degree"],
                coeff_mod_bit_sizes=encryption_config[encryption_method]["coeff_mod_bit_sizes"]
            )
            private_context.generate_galois_keys()
            private_context.generate_relin_keys()
            private_context.global_scale = 1 << encryption_config[encryption_method][
                "global_scale_bit_size"]

            serialized_public_context = private_context.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True
            )
            public_context = ts.context_from(serialized_public_context)
        elif encryption_method == "paillier":
            num_cores = -1 if encryption_config[encryption_method]["parallelize_on"] else 1
            private_context = Paillier.context(encryption_config[encryption_method]["key_bit_size"],
                                               djn_on=encryption_config[encryption_method]["djn_on"])
            paillier_key = private_context.to_public().serialize()
            public_context = Paillier.context_from(paillier_key)

        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=['node-1']
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=['node-2', 'node-3']
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_assist_trainer", return_value="assist_trainer"
        )
        mocker.patch.object(
            DualChannel, "send", return_value=0
        )

        vlr = VerticalLinearRegressionLabelTrainer(conf)
        vlr_t = VerticalLinearRegressionTrainer(conf_t)

        assert len(vlr.train_dataloader) == int(len(vlr.train) / vlr.batch_size) + 1

        for batch_idx, (_x_batch, _) in enumerate(vlr_t.train_dataloader):
            x_batch = _x_batch
            break
        for batch_idx, (_x_batch, _y_batch, _) in enumerate(vlr.train_dataloader):
            x_batch_label = _x_batch
            y_batch = _y_batch
            break
        for batch_idx, (_x_batch, _) in enumerate(vlr_t.val_dataloader):
            x_batch_val = _x_batch
            break

        pred_trainer = vlr_t.model(x_batch)
        pred_trainer_val = vlr_t.model(x_batch_val)
        pred_label_trainer = vlr.model(x_batch_label)

        loss_trainer = (pred_trainer ** 2).sum() / 2
        pred_residual = pred_label_trainer - y_batch
        loss_label_trainer = (pred_residual ** 2).sum() / 2
        loss = loss_trainer + loss_label_trainer
        d = pred_trainer + pred_residual

        if encryption_method == "paillier":
            en_pred_trainer_p = Paillier.serialize(Paillier.encrypt(public_context,
                                                                    pred_trainer.numpy().astype(np.float32).flatten(),
                                                                    precision=encryption_config[encryption_method][
                                                                        "precision"],
                                                                    obfuscation=True, num_cores=num_cores))
            en_loss_trainer_p = Paillier.serialize(Paillier.encrypt(public_context,
                                                                    float(loss_trainer),
                                                                    precision=encryption_config[encryption_method][
                                                                        "precision"],
                                                                    obfuscation=True, num_cores=num_cores))

        def mock_dual_label_t_recv(*args, **kwargs):
            if encryption_method == "ckks":
                if mock_label_t_recv.call_count == 1:
                    return ts.ckks_vector(public_context, pred_trainer.numpy().astype(np.float32).flatten()).serialize()
                elif mock_label_t_recv.call_count == 2:
                    return ts.ckks_vector(public_context, loss_trainer.numpy().astype(np.float32).flatten()).serialize()
                elif mock_label_t_recv.call_count == 3:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 4:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 5:
                    tmp = ("node-2", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp
            elif encryption_method == "paillier":
                if mock_label_t_recv.call_count == 1:
                    return en_pred_trainer_p
                elif mock_label_t_recv.call_count == 2:
                    return en_loss_trainer_p
                elif mock_label_t_recv.call_count == 3:
                    tmp = Paillier.ciphertext_from(public_context, en_pred_trainer_p)
                    return Paillier.serialize(np.sum(tmp * pred_trainer.numpy().astype(np.float32).flatten()))
                elif mock_label_t_recv.call_count == 4:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 5:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 6:
                    tmp = ("node-2", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp
            elif encryption_method == "plain":
                if mock_label_t_recv.call_count == 1:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 2:
                    return loss_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 3:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 4:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv.call_count == 5:
                    tmp = ("node-2", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp

        def mock_dual_label_t_recv_1(*args, **kwargs):
            if encryption_method == "ckks":
                if mock_label_t_recv_1.call_count == 1:
                    return ts.ckks_vector(public_context, pred_trainer.numpy().astype(np.float32).flatten()).serialize()
                elif mock_label_t_recv_1.call_count == 2:
                    return ts.ckks_vector(public_context, loss_trainer.numpy().astype(np.float32).flatten()).serialize()
                elif mock_label_t_recv_1.call_count == 3:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 4:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 5:
                    tmp = ("node-3", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp
            elif encryption_method == "paillier":
                if mock_label_t_recv_1.call_count == 1:
                    return en_pred_trainer_p
                elif mock_label_t_recv_1.call_count == 2:
                    return en_loss_trainer_p
                elif mock_label_t_recv_1.call_count == 3:
                    tmp = Paillier.ciphertext_from(public_context, en_pred_trainer_p)
                    return Paillier.serialize(np.sum(tmp * pred_trainer.numpy().astype(np.float32).flatten()))
                elif mock_label_t_recv_1.call_count == 4:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 5:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 6:
                    tmp = ("node-3", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp
            elif encryption_method == "plain":
                if mock_label_t_recv_1.call_count == 1:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 2:
                    return loss_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 3:
                    return pred_trainer.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 4:
                    return pred_trainer_val.numpy().astype(np.float32).flatten()
                elif mock_label_t_recv_1.call_count == 5:
                    tmp = ("node-3", vlr_t.model.state_dict()["linear.weight"][0])
                    return tmp

        mock_label_t_recv = mocker.patch.object(
            vlr.dual_channels["intermediate_label_trainer"]["node-2"], "recv", side_effect=mock_dual_label_t_recv
        )
        mock_label_t_recv_1 = mocker.patch.object(
            vlr.dual_channels["intermediate_label_trainer"]["node-3"], "recv", side_effect=mock_dual_label_t_recv_1
        )

        def mock_broadcast_recv(*args, **kwargs):
            if vlr.encryption_method == "ckks":
                return serialized_public_context
            elif vlr.encryption_method == "paillier":
                return private_context.to_public().serialize()

        def mock_broadcast_collect(*args, **kwargs):
            return [2, 2]

        mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_broadcast_recv
        )
        mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_broadcast_collect
        )

        def mock_gradients_loss(*args, **kwargs):
            if mock_gradients_loss_label.call_count == 1:
                return loss
            elif mock_gradients_loss_label.call_count == 2:
                tmp_w = vlr.model.linear.weight
                tmp_b = vlr.model.linear.bias
                return {"noised_gradient_label_trainer_w": tmp_w, "noised_gradient_label_trainer_b": tmp_b}

        mock_gradients_loss_label = mocker.patch.object(
            vlr.dual_channels["gradients_loss"], "recv", side_effect=mock_gradients_loss
        )
        # fit label_trainer
        vlr.fit()

        # mock for trainer
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        def mock_trainer_collect(*args, **kwargs):
            return [en_pred_trainer_p]

        mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_trainer_collect
        )

        def mock_dual_recv(*args, **kwargs):
            if mock_trainer_dual_recv.call_count == 1:
                if encryption_method == "ckks":
                    return ts.ckks_vector(public_context, d.numpy().astype(np.float32).flatten()).serialize()
                elif encryption_method == "paillier":
                    return Paillier.serialize(Paillier.encrypt(public_context, d.numpy().astype(np.float32).flatten(),
                                                               precision=encryption_config[encryption_method][
                                                                   "precision"],
                                                               obfuscation=True, num_cores=num_cores))
                elif encryption_method == "plain":
                    return d
            elif mock_trainer_dual_recv.call_count == 2:
                return [False, True, vlr.early_stopping_config["patience"]]

        mock_trainer_dual_recv = mocker.patch.object(
            vlr_t.dual_channels["intermediate_label_trainer"], "recv", side_effect=mock_dual_recv
        )

        def mock_gradients_trainer(*args, **kwargs):
            return vlr_t.model.linear.weight

        mocker.patch.object(
            vlr_t.dual_channels["gradients_loss"], "recv", side_effect=mock_gradients_trainer
        )

        def mock_broadcast_trainer_recv(*args, **kwargs):
            if encryption_method == "ckks":
                return serialized_public_context
            elif encryption_method == "paillier":
                return private_context.to_public().serialize()

        mocker.patch.object(
            vlr_t.broadcast_channel, "recv", side_effect=mock_broadcast_trainer_recv
        )
        # fit vlr_t
        vlr_t.fit()

        # mock for assist_trainer
        def mock_dual_init_recv(*args, **kwargs):
            if mock_dual_init_recv_.call_count == 1:
                return 1
            elif mock_dual_init_recv_.call_count == 2:
                return 1
            elif mock_dual_init_recv_.call_count == 3:
                return 1000
            elif mock_dual_init_recv_.call_count == 4:
                return encryption_config
            elif mock_dual_init_recv_.call_count == 5:
                return encryption_method

        mock_dual_init_recv_ = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_dual_init_recv
        )
        vlr_a = VerticalLinearRegressionAssistTrainer()

        if encryption_method == "paillier":
            num_cores = -1 if encryption_config[encryption_method]["parallelize_on"] else 1
            public_context = Paillier.context_from(vlr_a.public_context_ser)
        elif encryption_method == "ckks":
            public_context = ts.context_from(vlr_a.public_context_ser)

        def mock_dual_label_t_recv(*args, **kwargs):
            if mock_dual_label_recv_.call_count == 1:
                if encryption_method == "ckks":
                    return ts.ckks_vector(public_context, loss.numpy().astype(np.float32).flatten()
                                          ).serialize()
                elif encryption_method == "paillier":
                    return Paillier.serialize(Paillier.encrypt(public_context, float(loss),
                                                               precision=encryption_config[encryption_method][
                                                               "precision"], obfuscation=True, num_cores=num_cores))
                elif encryption_method == "plain":
                    return loss
            elif mock_dual_label_recv_.call_count == 2:
                if encryption_method == "ckks":
                    return ts.ckks_vector(public_context, vlr.model.linear.weight.numpy().astype(np.float32).flatten()
                                          ).serialize()
                elif encryption_method == "paillier":
                    return Paillier.serialize(Paillier.encrypt(public_context, vlr.model.linear.weight.numpy().astype(
                        np.float32).flatten(), precision=encryption_config[encryption_method]["precision"],
                                                               obfuscation=True, num_cores=num_cores))
            elif mock_dual_label_recv_.call_count == 3:
                if encryption_method == "ckks":
                    return ts.ckks_vector(public_context, vlr.model.linear.bias.numpy().astype(np.float32).flatten()
                                          ).serialize()
                elif encryption_method == "paillier":
                    return Paillier.serialize(Paillier.encrypt(public_context, vlr.model.linear.bias.numpy().astype(
                        np.float32).flatten(), precision=encryption_config[encryption_method]["precision"],
                                                               obfuscation=True, num_cores=num_cores))

        mock_dual_label_recv_ = mocker.patch.object(
            vlr_a.dual_channels["gradients_loss"]['node-1'], "recv", side_effect=mock_dual_label_t_recv
        )

        def mock_dual_trainer_t_recv(*args, **kwargs):
            if encryption_method == "ckks":
                return ts.ckks_vector(public_context, vlr_t.model.linear.weight.numpy().astype(np.float32).flatten()
                                      ).serialize()
            elif encryption_method == "paillier":
                return Paillier.serialize(Paillier.encrypt(public_context, vlr_t.model.linear.weight.numpy().astype(
                    np.float32).flatten(), precision=encryption_config[encryption_method]["precision"],
                                                           obfuscation=True, num_cores=num_cores))

        mocker.patch.object(
            vlr_a.dual_channels["gradients_loss"]['node-2'], "recv", side_effect=mock_dual_trainer_t_recv
        )
        mocker.patch.object(
            vlr_a.dual_channels["gradients_loss"]['node-3'], "recv", side_effect=mock_dual_trainer_t_recv
        )
        # fit assist_trainer
        vlr_a.fit()

        assert os.path.exists("/opt/checkpoints/unit_test/vertical_linear_regression_[STAGE_ID].pt")
        assert os.path.exists("/opt/checkpoints/unit_test/linear_reg_metric_train_[STAGE_ID].csv")
        assert os.path.exists("/opt/checkpoints/unit_test/linear_reg_metric_val_[STAGE_ID].csv")
        assert os.path.exists("/opt/checkpoints/unit_test/linear_reg_prediction_train_[STAGE_ID].csv")
        assert os.path.exists("/opt/checkpoints/unit_test/linear_reg_prediction_val_[STAGE_ID].csv")
        assert os.path.exists("/opt/checkpoints/unit_test/linear_reg_feature_importance_[STAGE_ID].csv")

        feature_importance = pd.read_csv("/opt/checkpoints/unit_test/linear_reg_feature_importance_[STAGE_ID].csv")
        assert len(feature_importance) == 5
        train_metric = pd.read_csv("/opt/checkpoints/unit_test/linear_reg_metric_train_[STAGE_ID].csv")
        assert len(train_metric.columns) == 6
        val_metric = pd.read_csv("/opt/checkpoints/unit_test/linear_reg_metric_val_[STAGE_ID].csv")
        assert len(val_metric.columns) == 6
