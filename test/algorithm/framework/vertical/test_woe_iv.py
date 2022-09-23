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
import random
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from algorithm.framework.vertical.binning_woe_iv.label_trainer import \
    VerticalBinningWoeIvLabelTrainer
from algorithm.framework.vertical.binning_woe_iv.trainer import \
    VerticalBinningWoeIvTrainer
from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier
from common.communication.gRPC.python.commu import Commu

def prepare_data():
    case_df = pd.DataFrame({
        'x0': list(np.random.random(800))+[999]*200,
        'x1': [0] * 500 + [1] * 500,
        'x2': [999] * 1000,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': [1] * 1000
    })
    case_df = case_df.astype("float32")
    y = [0] * 700 + [1] * 300
    random.shuffle(y)
    case_df['y'] = y
    case_df[['y', 'x0', 'x1', 'x2']].reset_index().rename(columns={'index': 'id'}).to_csv(
        "/opt/dataset/unit_test/breast_cancer_wisconsin_guest_train.csv", index=False
    )
    case_df[['x3', 'x4']].reset_index().rename(columns={'index': 'id'}).to_csv(
        "/opt/dataset/unit_test/breast_cancer_wisconsin_host_train.csv", index=False
    )
    case_df[['y', 'x0', 'x1', 'x2']].reset_index().rename(columns={'index': 'id'}).iloc[:100].to_csv(
        "/opt/dataset/unit_test/breast_cancer_wisconsin_guest_test.csv", index=False
    )
    case_df[['x3', 'x4']].reset_index().rename(columns={'index': 'id'}).iloc[:100].to_csv(
        "/opt/dataset/unit_test/breast_cancer_wisconsin_host_test.csv", index=False
    )
    case_df.reset_index().rename(columns={'index': 'id'}).to_csv(
        "/opt/dataset/unit_test/data.csv", index=False
    )


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/vertical_binning_woe_iv/label_trainer.json") as f:
        label_trainer_conf = json.load(f)
        label_trainer_conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        label_trainer_conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        label_trainer_conf["output"]["trainset"]["path"] = "/opt/checkpoints/unit_test_1"
        label_trainer_conf["output"]["valset"]["path"] = "/opt/checkpoints/unit_test_1"
        label_trainer_conf["input"]["trainset"][0]["name"] = "breast_cancer_wisconsin_guest_train.csv"
        label_trainer_conf["input"]["valset"][0]["name"] = "breast_cancer_wisconsin_guest_test.csv"
    yield label_trainer_conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/vertical_binning_woe_iv/trainer.json") as f:
        trainer_conf = json.load(f)
        trainer_conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        trainer_conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        trainer_conf["output"]["trainset"]["path"] = "/opt/checkpoints/unit_test_1"
        trainer_conf["output"]["valset"]["path"] = "/opt/checkpoints/unit_test_1"
        trainer_conf["input"]["trainset"][0]["name"] = "breast_cancer_wisconsin_host_train.csv"
        trainer_conf["input"]["valset"][0]["name"] = "breast_cancer_wisconsin_host_test.csv"
    yield trainer_conf


@pytest.fixture(scope="module", autouse=True)
def env():
    Commu.node_id="node-1"
    Commu.trainer_ids = ['node-1', 'node-2']
    Commu.scheduler_id = 'assist_trainer'
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
    if os.path.exists("/opt/checkpoints/unit_test_1"):
        shutil.rmtree("/opt/checkpoints/unit_test_1")


def simu_data():
    case_df = pd.read_csv("/opt/dataset/unit_test/data.csv", index_col='id').reset_index(drop=True)
    return case_df


class TestBinningWoeIv:
    @pytest.mark.parametrize("encryption_method, strategy, binning", [
        ("paillier", "mean", "equalWidth"), ("plain", "constant", "equalWidth"),
        ("plain", "mean", "equalFrequency")])
    def test_trainer(self, get_trainer_conf, encryption_method, strategy, binning, mocker):
        case_df = simu_data()
        train_conf = get_trainer_conf
        train_conf = get_trainer_conf
        if binning == "equalFrequency":
            train_conf["train_info"]["params"]['binning_params']['method'] = "equalFrequency"
        bwi = VerticalBinningWoeIvTrainer(train_conf)

        if encryption_method == "plain":
            bwi.train_params["encryption_params"] = {
                "method": "plain"
            }
        encryption_config = bwi.train_params["encryption_params"]

        if encryption_method == "paillier":
            pri_context = Paillier.context(encryption_config["key_bit_size"], djn_on=encryption_config["djn_on"])
        elif encryption_method == "plain":
            pass

        def mock_recv(*args, **kwargs):
            if encryption_method == "paillier":
                if mock_channel_recv.call_count <= 1:
                    return pri_context.to_public().serialize()
                elif mock_channel_recv.call_count % 2 == 0:
                    num_cores = -1 if encryption_config["parallelize_on"] else 1
                    label = case_df[["y"]].to_numpy().flatten().astype(np.int32)
                    en_label = Paillier.encrypt(pri_context,
                                                label,
                                                precision=encryption_config["precision"],
                                                obfuscation=True,
                                                num_cores=num_cores)
                    return Paillier.serialize(en_label)
            elif encryption_method == "plain":
                return case_df[["y"]]

        mock_channel_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_recv
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )
        bwi.fit()

    @pytest.mark.parametrize("encryption_method, strategy, binning", [
        ("paillier", "mean", "equalWidth"), ("plain", "constant", "equalWidth"),
        ("plain", "mean", "equalFrequency")])
    def test_label_trainer(self, get_label_trainer_conf, encryption_method, strategy, binning, mocker):
        label_train_conf = get_label_trainer_conf
        if binning == "equalFrequency":
            label_train_conf["train_info"]["params"]['binning_params']['method'] = "equalFrequency"
        mocker.patch.object(
            BroadcastChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        bwi = VerticalBinningWoeIvLabelTrainer(label_train_conf)
        bwi.broadcast_channel.remote_ids = ["node-2"]

        if encryption_method == "plain":
            bwi.train_params["encryption_params"] = {
                "method": "plain"
            }
        encryption_config = bwi.train_params["encryption_params"]

        if encryption_method == "paillier":
            pri_context = Paillier.context(encryption_config["key_bit_size"], djn_on=encryption_config["djn_on"])
            pub_context = Paillier.context_from(pri_context.to_public().serialize())
        elif encryption_method == "plain":
            pass

        def mock_collect(*args, **kwargs):
            case_df = simu_data()
            y = case_df[["y"]]
            case_df = case_df[['x3', 'x4']]
            bin_num = bwi.train_params["binning_params"]["bins"]
            labels = [i for i in range(bin_num)]
            columns_name = case_df.columns
            if bwi.train_params["binning_params"]['method'] == "equalWidth":
                case_df = pd.Series(case_df.columns).apply(
                    lambda x: pd.cut(case_df[x], bin_num, retbins=True, labels=labels)[0]).T
            else:
                case_df = pd.Series(case_df.columns).apply(
                    lambda x: pd.qcut(case_df[x], bin_num, retbins=True, duplicates='drop')[0]).T
                for i in case_df.columns:
                    case_df[i] = LabelEncoder().fit_transform(case_df[i])
            case_df.columns = columns_name

            if encryption_method == "paillier":
                # num_cores = -1 if encryption_config["parallelize_on"] else 1
                # label = y.to_numpy().flatten().astype(np.int32)
                # en_label = Paillier.encrypt(pri_context,
                #                             label,
                #                             precision=encryption_config["precision"],
                #                             obfuscation=True,
                #                             num_cores=num_cores)
                # encrypt_id_label_pair = Paillier.serialize(en_label)
                # en_label = Paillier.ciphertext_from(pub_context, encrypt_id_label_pair)
                # encrypt_id_label_pair = pd.DataFrame(en_label).rename(columns={0: 'y'})
                # tmp = []
                #
                # for feat in case_df.columns:
                #     feature_df = encrypt_id_label_pair.join(case_df[feat])
                #     tmp.append(feature_df.groupby([feat])['y'].agg({'count', 'sum'}))
                # bins_count = dict(zip(case_df.columns, [i['count'] for i in tmp]))
                # woe_feedback_list = dict(zip(case_df.columns, [i['sum'] for i in tmp]))
                # for _, feature in woe_feedback_list.items():
                #     woe_feedback_list[_] = feature.apply(lambda x: x.serialize())
                return [{"woe_feedback_list": {}, "bins_count": {}}]
            elif encryption_method == "plain":
                encrypt_id_label_pair = pd.DataFrame(y)
                tmp = []
                for feat in case_df.columns:
                    feature_df = encrypt_id_label_pair.join(case_df[feat])
                    tmp.append(feature_df.groupby([feat])['y'].agg({'count', 'sum'}))
                bins_count = dict(zip(case_df.columns, [i['count'] for i in tmp]))
                woe_feedback_list = dict(zip(case_df.columns, [i['sum'] for i in tmp]))
                return [{"woe_feedback_list": woe_feedback_list, "bins_count": bins_count}]

        mocker.patch.object(
            BroadcastChannel, "collect", side_effect=mock_collect
        )
        bwi.fit()

        # 检查是否正常留存
        assert os.path.exists("/opt/checkpoints/unit_test_1/vertical_binning_woe_iv_train.json")

        with open("/opt/checkpoints/unit_test_1/vertical_binning_woe_iv_train.json", "r", encoding='utf-8') as f:
            conf = json.loads(f.read())
            for k in ["woe", "iv", "count_neg", "count_pos", "ratio_pos", "ratio_neg"]:
                assert k in conf
