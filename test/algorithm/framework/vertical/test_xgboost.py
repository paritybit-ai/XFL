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
import random
import shutil
import string

from multiprocess.pool import ApplyResult

import numpy as np
import pandas as pd
import pytest

import service.fed_config
from algorithm.core.paillier_acceleration import embed, umbed
from algorithm.core.tree.tree_structure import Node, Tree
from algorithm.framework.vertical.xgboost import (decision_tree_label_trainer,
                                                  decision_tree_trainer)
from algorithm.framework.vertical.xgboost.label_trainer import \
    VerticalXgboostLabelTrainer
from algorithm.framework.vertical.xgboost.trainer import VerticalXgboostTrainer
from common.communication.gRPC.python.channel import (BroadcastChannel,
                                                      DualChannel)
from common.communication.gRPC.python.commu import Commu
from common.crypto.paillier.paillier import Paillier
from service.fed_config import FedConfig
from service.fed_node import FedNode
from algorithm.core.tree.tree_structure import Node, SplitInfo

random.seed(1)

private_context = Paillier.context(2048, True)
public_context = private_context.to_public()


def prepare_data():
    case_df = pd.DataFrame({
        'x0': np.arange(100),
        'x1': np.arange(100),
        'x2': 2 * np.arange(100) - 40.0,
        'x3': 3 * np.arange(100) + 1.0,
        'x4': np.arange(100)[::-1]
    })
    case_df['y'] = np.where(
        case_df['x0'] + case_df['x2'] + case_df['x3'] > 40, 1, 0)
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
    FedNode.node_id = "node-1"
    FedNode.node_name = "node-1"


def prepare_test_data():
    case_df = pd.DataFrame({
        'x0': np.arange(99),
        'x1': np.arange(99),
        'x2': 2 * np.arange(99) - 40.0,
        'x3': 3 * np.arange(99) + 1.0,
        'x4': np.arange(99)[::-1]
    })
    case_df['y'] = np.where(
        case_df['x0'] + case_df['x2'] + case_df['x3'] > 40, 1, 0)
    case_df[['y', 'x0', 'x1', 'x2']].to_csv(
        "/opt/dataset/unit_test/infer_guest.csv", index=True
    )
    case_df[['x3', 'x4']].to_csv(
        "/opt/dataset/unit_test/infer_host.csv", index=True
    )

    xgb_output = {
        "suggest_threshold": 0.6161117553710938,
        "lr": [0.3],
        "max_depth": [2],
        "trees": [
            {
                "party_id": "node-1",
                "tree_index": 0,
                "root_node_id": "0_4lN0P7QTwWq25Eei",
                "nodes": {
                    "0_4lN0P7QTwWq25Eei": {
                            "id": "0_4lN0P7QTwWq25Eei", "depth": 0, "left_node_id": "0_gw94EBW5tiD8kCqG",
                            "right_node_id": "0_vpKZWumTxYcojXLq",
                        "split_info": {
                            "owner_id": "node-1", "feature_idx": 0, "is_category": True,
                                        "split_point": None, "left_cat": [4, 2, 6, 1]
                        },
                        "is_leaf": False,
                        "weight": None, "linkage": None
                    }, "0_gw94EBW5tiD8kCqG": {
                        "id": "0_gw94EBW5tiD8kCqG", "depth": 1, "left_node_id": None, "right_node_id": None,
                        "split_info": None,
                        "is_leaf": True, "weight": 1.5769230769230769, "linkage": "left"
                    },
                    "0_vpKZWumTxYcojXLq": {
                        "id": "0_vpKZWumTxYcojXLq", "depth": 1, "left_node_id": None,
                        "right_node_id": None,
                        "split_info": None, "is_leaf": True, "weight": -1.5, "linkage": "right"
                    }
                }
            }
        ],
        "version": "1.0", "loss_method": "BCEWithLogitsLoss", "num_trees": 1,
        "node_id_group": {
            "0_4lN0P7QTwWq25Eei": ["0_4lN0P7QTwWq25Eei"]
        }
    }

    with open("/opt/checkpoints/unit_test/node-1/vertical_xgboost_guest.pmodel", 'w') as f:
        json.dump(xgb_output, f)

    xgb_output = {"4_WTqDQjPt39iMc7Ug": {"id": "4_WTqDQjPt39iMc7Ug",
                                         "split_info": {"owner_id": "node-2", "feature_idx": 0, "is_category": True,
                                                        "split_point": None, "left_cat": [1, 0, 2, 5]}}}
    with open("/opt/checkpoints/unit_test/node-2/vertical_xgboost_host.pmodel", 'w') as f:
        json.dump(xgb_output, f)


def enc_grad_hess(grad, hess):
    if grad is None:
        return Paillier.encrypt(context=private_context,
                                data=hess,
                                precision=7,
                                obfuscation=True,
                                num_cores=1)
    elif hess is None:
        return Paillier.encrypt(context=private_context,
                                data=grad,
                                precision=7,
                                obfuscation=True,
                                num_cores=1)
    else:
        grad_hess = embed([grad, hess], interval=(
            1 << 128), precision=64)
        enc_grad_hess = Paillier.encrypt(context=private_context,
                                         data=grad_hess,
                                         precision=0,  # must be 0
                                         obfuscation=True,
                                         num_cores=1)
        return enc_grad_hess


@pytest.fixture()
def get_label_trainer_infer_conf():
    conf = {
        "identity": "label_trainer",
        "model_info": {
            "name": "vertical_xgboost",
            "config": {}
        },
        "inference": True,
        "input": {
            "testset": [
                {
                    "type": "csv",
                    "path": "/opt/dataset/unit_test",
                    "name": "infer_guest.csv",
                    "has_label": True,
                    "has_id": True
                }
            ],
            "pretrained_model": {
                "path": "/opt/checkpoints/unit_test/node-1",
                "name": "vertical_xgboost_guest.pmodel"
            }
        },
        "output": {
            "path": "/opt/checkpoints/unit_test/node-1",
            "testset": {
                "name": "predicted_probabilities_train.csv"
            }
        },
        "train_info": {
            "interaction_params": {
            },
            "train_params": {
                "batch_size_val": 99
            }
        }
    }
    yield conf


@pytest.fixture()
def get_trainer_infer_conf():
    conf = {
        "identity": "trainer",
        "model_info": {
            "name": "vertical_xgboost",
            "config": {}
        },
        "inference": True,
        "input": {
            "testset": [
                {
                    "type": "csv",
                    "path": "/opt/dataset/unit_test",
                    "name": "infer_host.csv",
                    "has_label": False,
                    "has_id": True
                }
            ],
            "pretrained_model": {
                "path": "/opt/checkpoints/unit_test/node-2",
                "name": "vertical_xgboost_host.pmodel"
            }
        },
        "output": {
        },
        "train_info": {
            "interaction_params": {
            },
            "train_params": {
                "batch_size_val": 99
            }
        }
    }
    yield conf


@pytest.fixture()
def get_label_trainer_conf():
    with open("algorithm/config/vertical_xgboost/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_guest.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_guest.csv"
        conf["input"]["testset"] = []
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["prediction_train"]["name"] = "/opt/checkpoints/unit_test/predicted_probabilities_train.csv"
        conf["output"]["prediction_val"]["name"] = "/opt/checkpoints/unit_test/predicted_probabilities_val.csv"
        conf["output"]["model"]["name"] = "vertical_xgboost_guest.model"
        conf["output"]["proto_model"]["name"] = "vertical_xgboost_guest.pmodel"
        conf["output"]["feature_importance"]["name"] = "/opt/checkpoints/unit_test/feature_importances.csv"
        conf["train_info"]["train_params"]["num_bins"] = 10
        conf["train_info"]["train_params"]["max_depth"] = 2
        conf["train_info"]["train_params"]["min_sample_split"] = 1
        conf["train_info"]["train_params"]["downsampling"]["row"]["top_rate"] = 0.5
        conf["train_info"]["train_params"]["downsampling"]["row"]["other_rate"] = 0.5
        conf["train_info"]["train_params"]["num_trees"] = 1
        conf["train_info"]["train_params"]["max_num_cores"] = 2
        conf["train_info"]["train_params"]["metric"] = {
            "acc": {},
            "precision": {},
            "recall": {},
            "f1_score": {},
            "auc": {},
        }
        conf["train_info"]["train_params"]["early_stopping"]["key"] = "acc"
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("algorithm/config/vertical_xgboost/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_host.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_host.csv"
        conf["input"]["testset"] = []
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["model"]["name"] = "vertical_xgboost_host.model"
        conf["output"]["proto_model"]["name"] = "vertical_xgboost_host.pmodel"
        conf["train_info"]["train_params"]["num_bins"] = 10
        conf["train_info"]["train_params"]["max_depth"] = 2
        conf["train_info"]["train_params"]["min_sample_split"] = 1
        conf["train_info"]["train_params"]["num_trees"] = 1
        conf["train_info"]["train_params"]["max_num_cores"] = 2
        conf["train_info"]["train_params"]["advanced"]["row_batch"] = 20
    yield conf


@pytest.fixture(scope="module", autouse=True)
def env():
    Commu.node_id = "node-1"
    Commu.trainer_ids = ['node-1', 'node-2']
    Commu.scheduler_id = 'assist_trainer'
    os.chdir("python")
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test/node-1"):
        os.makedirs("/opt/checkpoints/unit_test/node-1")
    if not os.path.exists("/opt/checkpoints/unit_test/node-2"):
        os.makedirs("/opt/checkpoints/unit_test/node-2")
    if not os.path.exists("/opt/config/unit_test"):
        os.makedirs("/opt/config/unit_test")
    prepare_data()
    prepare_test_data()
    yield
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/config/unit_test"):
        shutil.rmtree("/opt/config/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")
    os.chdir("..")


class TestVerticalXgboost:
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.parametrize('embed', [(True), (False)])
    def test_label_trainer(self, get_label_trainer_conf, embed, mocker):

        def mock_generate_id(*args, **kwargs):
            return str(mock_tree_generate_id.call_count)

        def mock_dualchannel_recv(*args, **kwargs):
            if embed:
                # recv summed_grad_hess
                if mock_channel_recv.call_count in [1, 2, 4]:
                    hist_list = [(np.zeros(8), np.array([8] * 10))
                                 for _ in range(2)]
                    return [False, hist_list, [2]]
                elif mock_channel_recv.call_count in [6, 7]:
                    return {'1': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            '2': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            '3': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            }
                elif mock_channel_recv.call_count <= 5 or (
                        mock_channel_recv.call_count >= 8 and mock_channel_recv.call_count <= 12):
                    # features = pd.DataFrame({
                    #     'x3': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]),
                    #     'x4': np.array([9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                    # }

                    # )
                    # sample_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 48, 52, 53, 55, 59, 60, 61, 63, 64, 65, 66, 68, 70, 73, 74, 75, 76, 77, 78, 79]
                    # grad = [0.8333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.5, -0.5]
                    # hess = [0.41666666, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.25, 0.25]
                    # grad_hess = embed([grad, hess], interval=(
                    #     1 << 128), precision=64)
                    # enc_grad_hess = Paillier.encrypt(context=private_context,
                    #                                  data=grad_hess,
                    #                                  precision=0,  # must be 0
                    #                                  obfuscation=True,
                    #                                  num_cores=1)
                    # enc_grad_hess = Paillier.serialize(enc_grad_hess, compression=False)
                    # grad_hess = Paillier.ciphertext_from(public_context, enc_grad_hess, compression=False)
                    # big_feature = Feature.create(values=features.iloc[sample_index,:],sample_index=sample_index, grad_hess=grad_hess)
                    # res = []
                    # for col_name in big_feature.feature_columns:
                    #     res.append(big_feature.data.groupby([col_name])['xfl_grad_hess'].agg({'count', 'sum'}))
                    # hist_list = [(res_hist['sum'].to_numpy(), res_hist['count'].to_numpy()) for res_hist in res]
                    hist_list = [(np.zeros(8), np.array([8] * 10))
                                 for _ in range(2)]
                    return [False, hist_list]
                elif mock_channel_recv.call_count <= 7:
                    return {'1': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '2': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '3': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True])
                            }
                elif mock_channel_recv.call_count <= 14 and mock_channel_recv.call_count >= 13:
                    return {'1': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '2': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '3': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True])
                            }
            elif not embed:
                if mock_channel_recv.call_count in [1, 2, 4]:
                    hist_list = [(np.zeros(8), np.zeros(
                        8), np.array([8] * 10)) for _ in range(2)]
                    return [False, hist_list, [2]]
                elif mock_channel_recv.call_count in [6, 7]:
                    return {'1': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            '2': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            '3': np.packbits(np.array([True, True, True, True, True, True, True, True, True,
                                                       True, True, True, True, True, True, True, True, True,
                                                       True, True])),
                            }
                elif mock_channel_recv.call_count <= 5 or (
                        mock_channel_recv.call_count >= 8 and mock_channel_recv.call_count <= 12):
                    hist_list = [(np.zeros(8), np.zeros(
                        8), np.array([8] * 10)) for _ in range(2)]
                    return [False, hist_list]
                elif mock_channel_recv.call_count <= 7:
                    return {'1': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '2': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '3': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True])
                            }
                elif mock_channel_recv.call_count <= 14 and mock_channel_recv.call_count >= 13:
                    return {'1': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '2': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True]),
                            '3': np.array([True, True, True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True, True, True,
                                           True, True])
                            }

        def mock_broadcasetchannel_recv():
            pass

        if not embed:
            mocker.patch.object(decision_tree_label_trainer, "EMBEDING", False)

        mocker.patch("service.fed_control._send_progress")
        mocker.patch.object(FedConfig, "get_trainer", return_value=["node-2"])
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(Commu, "node_id", "node-1")
        mocker.patch.object(Commu, "trainer_ids", ["node-1", "node-2"])
        mocker.patch.object(Commu, "scheduler_id", "scheduler")
        mocker.patch.object(
            BroadcastChannel, "collect", return_value=[{"train": (80, 3), "valid": (20, 3)}]
        )

        mocker.patch.object(
            BroadcastChannel, "broadcast"
        )

        mocker.patch.object(
            DualChannel, "send"
        )
        mock_channel_recv = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_dualchannel_recv
        )
        mock_tree_generate_id = mocker.patch.object(
            Tree, "_generate_id", side_effect=mock_generate_id
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        mocker.patch.object(
            FedNode, "config",  return_value={"trainer": {"node-2": []}}
        )

        xgb_label_trainer = VerticalXgboostLabelTrainer(
            get_label_trainer_conf)
        
        mocker.patch.object(
            xgb_label_trainer.channels["sync"], "collect",  return_value=[{"node-2": [1, 2]}, {"node-3": [1, 2]}]
        )

        xgb_label_trainer.fit()

        self.check_label_trainer_output()

    @pytest.mark.parametrize('embed', [(True), (False)])
    def test_trainer(self, get_trainer_conf, embed, mocker):

        def mock_broadcastchannel_recv(*args, **kwargs):
            config = {
                "train_info": {
                    "interaction_params": {
                        "save_frequency": -1,
                        "echo_training_metrics": True,
                        "write_training_prediction": True,
                        "write_validation_prediction": True
                    },
                    "train_params": {
                        "lossfunc": {
                            "BCEWithLogitsLoss": {}
                        },
                        "num_trees": 1,
                        "num_bins": 10,
                        "downsampling": {
                            "row": {
                                "run_goss": True
                            }
                        },
                        "encryption": {
                            "paillier": {
                                "key_bit_size": 2048,
                                "precision": 7,
                                "djn_on": True,
                                "parallelize_on": True
                            }
                        },
                        "batch_size_val": 40960
                    }
                }
            }
            if broadchannel_recv_mocker.call_count == 1:
                return config
            # elif broadchannel_recv_mocker.call_count == 2:
            #     encryption = config["train_info"]["train_params"]["encryption"]
            #     if "paillier" in encryption:
            #         encryption = encryption["paillier"]
            #         private_context = Paillier.context(
            #             encryption["key_bit_size"], encryption["djn_on"])
            #         return private_context.to_public().serialize()
            #     else:
            #         return None

            if embed:
                # recv embed grad hess
                if broadchannel_recv_mocker.call_count in [3, 6]:
                    grad = np.array(
                        [0.8333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -
                         0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.5, -0.5])
                    hess = np.array(
                        [0.41666666, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.25, 0.25])
                    return Paillier.serialize(enc_grad_hess(grad, hess), compression=False)

                # recv public context for Paillier
                elif broadchannel_recv_mocker.call_count == 2:
                    return public_context.serialize()

                # recv tree node
                elif broadchannel_recv_mocker.call_count == 4:
                    def _generate_id():
                        id = ''.join(random.sample(
                            string.ascii_letters + string.digits, 16))
                        return id

                    return Node(id=_generate_id(), depth=0)
            elif not embed:
                # recv  grad and hess
                if broadchannel_recv_mocker.call_count in [3, 6]:
                    grad = np.array(
                        [0.8333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -
                         0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.5, -0.5])
                    hess = np.array(
                        [0.41666666, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666,
                         0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.41666666, 0.25, 0.25])
                    return Paillier.serialize(enc_grad_hess(grad, None), compression=False), Paillier.serialize(
                        enc_grad_hess(None, hess), compression=False)

                # recv public context for Paillier
                elif broadchannel_recv_mocker.call_count == 2:
                    return public_context.serialize()

                # recv tree node
                elif broadchannel_recv_mocker.call_count == 4:
                    def _generate_id():
                        id = ''.join(random.sample(
                            string.ascii_letters + string.digits, 16))
                        return id

                    return Node(id=_generate_id(), depth=0)

        def mock_dualchannel_recv(*args, **kwargs):
            # recv min split info
            if dualchannel_recv_mocker.call_count == 1:
                return -1, 1, 1

            # recv early stop
            elif dualchannel_recv_mocker.call_count == 2:
                return True

        if not embed:
            mocker.patch.object(decision_tree_trainer, "EMBEDING", False)

        mocker.patch.object(FedConfig, "get_label_trainer",
                            return_value=["node-1"])
        mocker.patch.object(FedNode, "node_id", "node-2")
        mocker.patch.object(FedNode, "create_channel")
        mocker.patch.object(Commu, "node_id", "node-2")
        mocker.patch.object(Commu, "trainer_ids", ["node-1", "node-2"])
        mocker.patch.object(Commu, "scheduler_id", "scheduler")

        broadchannel_recv_mocker = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_broadcastchannel_recv
        )
        dualchannel_recv_mocker = mocker.patch.object(
            DualChannel, "recv", side_effect=mock_dualchannel_recv
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        xgb_trainer = VerticalXgboostTrainer(get_trainer_conf)
        xgb_trainer.fit()
        self.check_trainer_output()

    @staticmethod
    def check_label_trainer_output():
        # 检查是否正确输出了预测值文件
        assert os.path.exists(
            "/opt/checkpoints/unit_test/predicted_probabilities_train.csv")
        assert os.path.exists(
            "/opt/checkpoints/unit_test/predicted_probabilities_val.csv")

        # 检查是否正确输出了模型文件
        assert os.path.exists(
            "/opt/checkpoints/unit_test/node-2/vertical_xgboost_host.pmodel")
        assert os.path.exists(
            "/opt/checkpoints/unit_test/node-1/vertical_xgboost_guest.pmodel")

        # 检查是否正确输出了model config
        assert os.path.exists("/opt/checkpoints/unit_test/model_config.json")
        with open("/opt/checkpoints/unit_test/model_config.json") as f:
            model_config = json.load(f)
        assert model_config[0]["class_name"] == "VerticalXGBooster"
        assert model_config[0]["filename"] == "vertical_xgboost_guest.pmodel"

        # 检查是否正确输出了feature importance文件
        assert os.path.exists(
            "/opt/checkpoints/unit_test/feature_importances.csv")

    @staticmethod
    def check_trainer_output():
        # 检查是否正确输出了模型文件
        assert os.path.exists(
            "/opt/checkpoints/unit_test/vertical_xgboost_host.pmodel")

        # 检查是否正确输出了model config
        assert os.path.exists("/opt/checkpoints/unit_test/model_config.json")
        with open("/opt/checkpoints/unit_test/model_config.json") as f:
            model_config = json.load(f)
        assert model_config[2]["class_name"] == "VerticalXGBooster"
        assert model_config[2]["filename"] == "vertical_xgboost_host.pmodel"
        
