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
import string
import pickle

import numpy as np
import pandas as pd
import pytest

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
from common.utils.tree_pickle_structure import NodePickle, TreePickle
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


def prepare_test_data():
    case_df = pd.DataFrame({
        'x0': np.arange(100),
        'x1': np.arange(100),
        'x2': 2 * np.arange(100) - 40.0,
        'x3': 3 * np.arange(100) + 1.0,
        'x4': np.arange(100)[::-1]
    })
    case_df['y'] = np.where(
        case_df['x0'] + case_df['x2'] + case_df['x3'] > 40, 1, 0)
    case_df[['y', 'x0', 'x1', 'x2']].to_csv(
        "/opt/dataset/unit_test/infer_guest.csv", index=True
    )
    case_df[['x3', 'x4']].to_csv(
        "/opt/dataset/unit_test/infer_host.csv", index=True
    )

    nodes = {
        "C6PR8y73s1qxt9Zc": Node(
            "C6PR8y73s1qxt9Zc", 1, None,
            "yJnqV0wsemk5fxgR", "8D5tWZ2GgmIYrACd", None,
            SplitInfo("node-1", 0, True, 30.0), False, -0.1, ""
        ),
        "yJnqV0wsemk5fxgR": Node(
            "yJnqV0wsemk5fxgR", 1, None,
            None, None, "C6PR8y73s1qxt9Zc",
            SplitInfo("node-1", 0, True, 10.0), True, 0.5, "left"
        ),
        "8D5tWZ2GgmIYrACd": Node(
            "8D5tWZ2GgmIYrACd", 1, None,
            None, None, "C6PR8y73s1qxt9Zc",
            SplitInfo("node-2", 0, True, 80.0), True, -0.2, "right"
        )
    }
    nodes_pkl = {
        "C6PR8y73s1qxt9Zc": NodePickle(
            "C6PR8y73s1qxt9Zc", 1, None, "yJnqV0wsemk5fxgR", "8D5tWZ2GgmIYrACd", False, -0.1, "", 30.0, 0, True,
            "node-1"
        ),
        "yJnqV0wsemk5fxgR": NodePickle(
            "yJnqV0wsemk5fxgR", 1, "C6PR8y73s1qxt9Zc", None, None, True, 0.5, "left", 10.0, 0, True, "node-1"
        ),
        "8D5tWZ2GgmIYrACd": NodePickle(
            "8D5tWZ2GgmIYrACd", 1, "C6PR8y73s1qxt9Zc", None, None, True, -0.2, "right", 80.0, 0, True, "node-2"
        )
    }
    tree = TreePickle("node-1", nodes, nodes["C6PR8y73s1qxt9Zc"], "C6PR8y73s1qxt9Zc")
    xgb_output = {
        "trees": [tree],
        "num_trees": 1,
        "lr": 0.3,
        "max_depth": 1,
        "suggest_threshold": 0.5,
        "node_id_group": {"C6PR8y73s1qxt9Zc": ["C6PR8y73s1qxt9Zc", "yJnqV0wsemk5fxgR", "8D5tWZ2GgmIYrACd"]}
    }
    with open("/opt/checkpoints/unit_test/node-1/vertical_xgboost_guest.pkl", 'wb') as f:
        pickle.dump(xgb_output, f)

    xgb_output = {
        "nodes": nodes_pkl,
        "num_trees": 1
    }
    with open("/opt/checkpoints/unit_test/node-2/vertical_xgboost_host.pkl", 'wb') as f:
        pickle.dump(xgb_output, f)


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
            "pretrain_model": {
                "path": "/opt/checkpoints/unit_test/node-1",
                "name": "vertical_xgboost_guest.pkl"
            }
        },
        "output": {
            "testset": {
                "type": "csv",
                "path": "/opt/checkpoints/unit_test/node-1",
                "name": "predicted_probabilities_train.csv"
            }
        },
        "train_info": {
            "device": "cpu",
            "interaction_params": {
            },
            "params": {
                "validation_batch_size": 99
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
            "pretrain_model": {
                "path": "/opt/checkpoints/unit_test/node-2",
                "name": "vertical_xgboost_host.pkl"
            }
        },
        "output": {
        },
        "train_info": {
            "device": "cpu",
            "interaction_params": {
            },
            "params": {
                "validation_batch_size": 99
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
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
        conf["train_info"]["params"]["num_bins"] = 10
        conf["train_info"]["params"]["min_sample_split"] = 1
        conf["train_info"]["params"]["top_rate"] = 0.5
        conf["train_info"]["params"]["other_rate"] = 0.3
        conf["train_info"]["params"]["num_trees"] = 1
        conf["train_info"]["params"]["max_num_cores"] = 2
        conf["train_info"]["params"]["row_batch"] = 20

        conf["train_info"]["params"]["metric_config"] = {
            "acc": {},
            "precision": {},
            "recall": {},
            "f1_score": {},
            "auc": {},
        }
        conf["train_info"]["params"]["early_stopping_params"]["key"] = "acc"
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("algorithm/config/vertical_xgboost/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_host.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_host.csv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["train_info"]["params"]["num_bins"] = 10
        conf["train_info"]["params"]["min_sample_split"] = 1
        conf["train_info"]["params"]["top_rate"] = 0.5
        conf["train_info"]["params"]["other_rate"] = 0.3
        conf["train_info"]["params"]["num_trees"] = 1
        conf["train_info"]["params"]["max_num_cores"] = 2
        conf["train_info"]["params"]["row_batch"] = 20
    yield conf


@pytest.fixture(scope="module", autouse=True)
def env():
    os.chdir("python")
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test/node-1"):
        os.makedirs("/opt/checkpoints/unit_test/node-1")
    if not os.path.exists("/opt/checkpoints/unit_test/node-2"):
        os.makedirs("/opt/checkpoints/unit_test/node-2")
    prepare_data()
    prepare_test_data()
    yield
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")
    os.chdir("..")


class TestVerticalXgboost:
    def test_predict_label_trainer(self, get_label_trainer_infer_conf, mocker):
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "recv", return_value={"8D5tWZ2GgmIYrACd": True}
        )
        xgb_label_trainer = VerticalXgboostLabelTrainer(get_label_trainer_infer_conf)
        xgb_label_trainer.predict()
        df = pd.read_csv("/opt/checkpoints/unit_test/node-1/predicted_probabilities_train.csv")
        assert (df["pred"] > 0.5).sum() == 30

    def test_predict_trainer(self, get_trainer_infer_conf, mocker):
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=0
        )
        xgb_label_trainer = VerticalXgboostTrainer(get_trainer_infer_conf)
        xgb_label_trainer.predict()
        assert not os.path.exists("/opt/checkpoints/unit_test/node-2/predicted_probabilities_train.csv")

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.parametrize('embed', [(True), (False)])
    def test_label_trainer(self, get_label_trainer_conf, embed, mocker):

        def mock_generate_id():
            return str(mock_tree_generate_id.call_count)

        def mock_dualchannel_recv(*args, **kwargs):
            if embed:
                # recv summed_grad_hess
                if mock_channel_recv.call_count <= 5 or (
                        mock_channel_recv.call_count >= 8 and mock_channel_recv.call_count <= 12):
                    hist_list = [(np.zeros(8), np.array([8] * 10)) for _ in range(2)]
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
                if mock_channel_recv.call_count <= 5 or (
                        mock_channel_recv.call_count >= 8 and mock_channel_recv.call_count <= 12):
                    hist_list = [(np.zeros(8), np.zeros(8), np.array([8] * 10)) for _ in range(2)]
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

        mocker.patch.object(FedConfig, "get_trainer", return_value=["node-2"])
        mocker.patch.object(FedNode, "node_id", "node-1")
        mocker.patch.object(Commu, "node_id", "node-1")
        mocker.patch.object(Commu, "trainer_ids", ["node-1", "node-2"])
        mocker.patch.object(Commu, "scheduler_id", "scheduler")

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

        xgb_label_trainer = VerticalXgboostLabelTrainer(
            get_label_trainer_conf)
        xgb_label_trainer.fit()

        self.check_label_trainer_output()

    @pytest.mark.parametrize('embed', [(True), (False)])
    def test_trainer(self, get_trainer_conf, embed, mocker):

        def mock_broadcastchannel_recv(*args, **kwargs):
            if embed:
                # recv embed grad hess
                if broadchannel_recv_mocker.call_count == 2:
                    grad = np.array(
                        [0.8333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333,
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
                elif broadchannel_recv_mocker.call_count == 1:
                    return public_context.serialize()

                # recv tree node
                elif broadchannel_recv_mocker.call_count == 3:
                    def _generate_id():
                        id = ''.join(random.sample(
                            string.ascii_letters + string.digits, 16))
                        return id

                    return Node(id=_generate_id(), depth=0)
            elif not embed:
                # recv  grad and hess
                if broadchannel_recv_mocker.call_count == 2:
                    grad = np.array(
                        [0.8333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                         -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.8333333, -0.8333333, -
                         0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333,
                         -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333, -0.8333333,
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
                elif broadchannel_recv_mocker.call_count == 1:
                    return public_context.serialize()

                # recv tree node
                elif broadchannel_recv_mocker.call_count == 3:
                    def _generate_id():
                        id = ''.join(random.sample(
                            string.ascii_letters + string.digits, 16))
                        return id

                    return Node(id=_generate_id(), depth=0)

        def mock_dualchannel_recv(*args, **kwargs):

            # recv min split info
            if dualchannel_recv_mocker.call_count == 1:
                return 1, 1

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
        xgb_trainer = VerticalXgboostTrainer(get_trainer_conf)
        xgb_trainer.fit()
        self.check_trainer_output()

    @staticmethod
    def check_label_trainer_output():
        assert os.path.exists(
            "/opt/checkpoints/unit_test/predicted_probabilities_train.csv")
        assert os.path.exists(
            "/opt/checkpoints/unit_test/predicted_probabilities_val.csv")

        assert os.path.exists(
            "/opt/checkpoints/unit_test/vertical_xgboost_guest_1.pkl")
        assert os.path.exists(
            "/opt/checkpoints/unit_test/vertical_xgboost_guest.pkl")

        assert os.path.exists("/opt/checkpoints/unit_test/model_config.json")
        with open("/opt/checkpoints/unit_test/model_config.json") as f:
            model_config = json.load(f)
        assert model_config[0]["class_name"] == "VerticalXGBooster"
        assert model_config[0]["filename"] == "vertical_xgboost_guest.pkl"

        assert os.path.exists(
            "/opt/checkpoints/unit_test/feature_importances.csv")

    @staticmethod
    def check_trainer_output():
        assert os.path.exists(
            "/opt/checkpoints/unit_test/vertical_xgboost_host.pkl")

        assert os.path.exists("/opt/checkpoints/unit_test/model_config.json")
        with open("/opt/checkpoints/unit_test/model_config.json") as f:
            model_config = json.load(f)
        assert model_config[2]["class_name"] == "VerticalXGBooster"
        assert model_config[2]["filename"] == "vertical_xgboost_host.pkl"
