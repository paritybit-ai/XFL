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

import os
import copy

import pytest
import json

from multiprocess.pool import ApplyResult

import pandas as pd
import numpy as np

import service.fed_config
from service.fed_config import FedConfig
from algorithm.core.paillier_acceleration import embed
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.communication.gRPC.python.commu import Commu
from common.crypto.paillier.paillier import Paillier
from algorithm.core.tree.tree_structure import Node
from algorithm.framework.vertical.xgboost.label_trainer import VerticalXgboostLabelTrainer
from algorithm.framework.vertical.xgboost.trainer import VerticalXgboostTrainer
from algorithm.framework.vertical.xgboost.decision_tree_label_trainer import VerticalDecisionTreeLabelTrainer
from algorithm.framework.vertical.xgboost.decision_tree_trainer import VerticalDecisionTreeTrainer


@pytest.fixture(scope='module', autouse=True)
def prepare_data(tmp_factory):
    df = pd.DataFrame({
        "x0": np.random.random(200),
        "x1": np.round(np.random.random(200) * 10.0),
        "x2": np.random.uniform(200) * 2.0,
        "x3": np.random.random(200) * 3.0,
        "x4": np.arange(0, 200, 1),
        'y': np.round(np.random.random(200))
    })
    df[['y', 'x0', 'x1', 'x2']].head(120).to_csv(
        tmp_factory.join("train_guest.csv"), index=True, index_label='id'
    )
    df[['y', 'x0', 'x1', 'x2']].tail(80).to_csv(
        tmp_factory.join("test_guest.csv"), index=True, index_label='id'
    )
    df[['x3', 'x4']].head(120).to_csv(
        tmp_factory.join("train_host.csv"), index=True, index_label='id'
    )
    df[['x3', 'x4']].tail(80).to_csv(
        tmp_factory.join("test_host.csv"), index=True, index_label='id'
    )
    Commu.node_id = "node-1"
    Commu.trainer_ids = ['node-1', 'node-2']


@pytest.fixture(scope='module', autouse=True)
def prepare_model(tmp_factory):
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
    with open(tmp_factory.join("vertical_xgboost_guest.model"), 'w') as f:
        json.dump(xgb_output, f)

    xgb_output = {"4_WTqDQjPt39iMc7Ug": {"id": "4_WTqDQjPt39iMc7Ug",
                                         "split_info": {"owner_id": "node-2", "feature_idx": 0, "is_category": True,
                                                        "split_point": None, "left_cat": [1, 0, 2, 5]}}}
    with open(tmp_factory.join("vertical_xgboost_host.model"), 'w') as f:
        json.dump(xgb_output, f)


@pytest.fixture()
def get_label_trainer_infer_conf(tmp_factory):
    conf = {
        "identity": "label_trainer",
        "model_info": {
            "name": "vertical_xgboost"
        },
        "inference": True,
        "input": {
            "testset": [
                {
                    "type": "csv",
                    "path": str(tmp_factory),
                    "name": "test_guest.csv",
                    "has_label": True,
                    "has_id": True
                }
            ],
            "pretrained_model": {
                "path": str(tmp_factory),
                "name": "vertical_xgboost_guest.model"
            }
        },
        "output": {
            "path": str(tmp_factory),
            "testset": {
                "name": "predicted_probabilities_train.csv"
            }
        },
        "train_info": {
            "interaction_params": {
            },
            "train_params": {
                "batch_size_val": 80
            }
        }
    }
    yield conf


@pytest.fixture()
def get_trainer_infer_conf(tmp_factory):
    conf = {
        "identity": "trainer",
        "model_info": {
            "name": "vertical_xgboost"
        },
        "inference": True,
        "input": {
            "testset": [
                {
                    "type": "csv",
                    "path": str(tmp_factory),
                    "name": "test_host.csv",
                    "has_label": False,
                    "has_id": True
                }
            ],
            "pretrained_model": {
                "path": str(tmp_factory),
                "name": "vertical_xgboost_host.model"
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


class TestVerticalXGBoost:
    @pytest.mark.parametrize("dataset, empty", [("train", False), ("valid", False), ("test", False),
                                                ("train", True), ("valid", True), ("test", True)])
    def test_check_dataset(self, get_label_trainer_infer_conf, mocker, tmp_factory, dataset, empty):
        with open("python/algorithm/config/vertical_xgboost/label_trainer.json") as f:
            conf = json.load(f)
            conf["input"]["trainset"][0]["path"] = str(tmp_factory)
            conf["input"]["trainset"][0]["name"] = "train_guest.csv"
            conf["input"]["valset"][0]["path"] = str(tmp_factory)
            conf["input"]["valset"][0]["name"] = "test_guest.csv"
            if dataset == "test":
                conf["inference"] = True
                conf["input"]["testset"][0]["path"] = str(tmp_factory)
                conf["input"]["testset"][0]["name"] = "test_guest.csv"
            else:
                del conf["input"]["testset"]
            conf["output"]["path"] = str(tmp_factory)

        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=None
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        xgb_label_trainer = VerticalXgboostLabelTrainer(conf)

        def mock_dim():
            if empty:
                d = {"train": (120, 2), "valid": (80, 2), "test": (80, 2)}
                d[dataset] = (d[dataset][0], 0)
                dims = [d]
            else:
                dims = [{"train": (120, 2), "valid": (80, 2), "test": (80, 2)}]
                d = {"train": (120, 2), "valid": (80, 2), "test": (80, 2)}
                d[dataset] = (1, 2)
                dims.append(d)
            return dims

        mocker.patch.object(
            xgb_label_trainer.channels["check_dataset_com"], "collect", side_effect=mock_dim
        )

        if empty:
            if dataset == "train":
                xgb_label_trainer.train_features = pd.DataFrame()
            elif dataset == "valid":
                xgb_label_trainer.val_features = pd.DataFrame()
            elif dataset == "test":
                xgb_label_trainer.test_features = pd.DataFrame()

        with pytest.raises(ValueError) as e:
            xgb_label_trainer.check_dataset()

    @pytest.mark.parametrize("num_bins", [2, 8, 128, 1024, 100000])
    def test_cat_label_trainer(self, mocker, tmp_factory, num_bins):
        """
        通过参数化测试当分类数大于小于num_bins的两种情况，同时测试num_bins在三段区间下的初始化
        Args:
                tmp_factory:
                num_bins:

        Returns:

        """
        with open("python/algorithm/config/vertical_xgboost/label_trainer.json") as f:
            conf = json.load(f)
            conf["input"]["trainset"][0]["path"] = str(tmp_factory)
            conf["input"]["trainset"][0]["name"] = "train_guest.csv"
            conf["input"]["valset"][0]["path"] = str(tmp_factory)
            conf["input"]["valset"][0]["name"] = "test_guest.csv"
            conf["output"]["path"] = str(tmp_factory)
            conf["train_info"]["train_params"]["category"]["cat_features"]["col_names"] = [
                "x1"]
            conf["train_info"]["train_params"]["num_bins"] = num_bins
            del conf["input"]["testset"]

        mocker.patch.object(
            BroadcastChannel, "__init__", return_value=None
        )
        
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=None
        )

        xgb_label_trainer = VerticalXgboostLabelTrainer(conf)
        mocker.patch.object(
            xgb_label_trainer.channels["check_dataset_com"], "collect", return_value=[]
        )
        xgb_label_trainer.fit()
        self.check_label_trainer_output(tmp_factory)

    # def test_decision_tree_trainer(self, mocker, tmp_factory):
    #     with open("python/algorithm/config/vertical_xgboost/trainer.json") as f:
    #         conf = json.load(f)
    #         conf["input"]["trainset"][0]["path"] = str(tmp_factory)
    #         conf["input"]["trainset"][0]["name"] = "train_host.csv"
    #         conf["input"]["valset"][0]["path"] = str(tmp_factory)
    #         conf["input"]["valset"][0]["name"] = "test_host.csv"
    #         del conf["input"]["testset"]
    #         conf["output"]["path"] = str(tmp_factory)
    #         # if conf["train_info"]["train_params"]["downsampling"]["row"]["run_goss"]:
    #         #     conf["train_info"]["train_params"]["downsampling"]["row"]["top_rate"] = 0.5
    #         #     conf["train_info"]["train_params"]["downsampling"]["row"]["other_rate"] = 0.5
    #         conf["train_info"]["train_params"]["category"]["cat_features"]["col_index"] = "1"

    #     # mocker channels in VerticalXgboostTrainer.__init__
    #     mocker.patch.object(
    #         DualChannel, "__init__", return_value=None
    #     )
    #     mocker.patch.object(
    #         BroadcastChannel, "send", return_value=None
    #     )
    #     mocker.patch.object(
    #         DualChannel, "send", return_value=None
    #     )

    #     def mock_func(*args, **kwargs):
    #         """
    #         mock encryption keys
    #         Args:
    #                 *args:
    #                 **kwargs:

    #         Returns:
    #                 the paillier context
    #         """
    #         config = {
    #             "train_info": {
    #                 "interaction_params": {
    #                     "save_frequency": -1,
    #                     "echo_training_metrics": True,
    #                     "write_training_prediction": True,
    #                     "write_validation_prediction": True
    #                 },
    #                 "train_params": {
    #                     "lossfunc": {
    #                         "BCEWithLogitsLoss": {}
    #                     },
    #                     "num_trees": 10,
    #                     "num_bins": 16,
    #                     "downsampling": {
    #                         "row": {
    #                             "run_goss": True
    #                         }
    #                     },
    #                     "encryption": {
    #                         "paillier": {
    #                             "key_bit_size": 2048,
    #                             "precision": 7,
    #                             "djn_on": True,
    #                             "parallelize_on": True
    #                         }
    #                     },
    #                     "batch_size_val": 40960
    #                 }
    #             }
    #         }
    #         if mock_broadcast_recv.call_count == 1:
    #             return config
    #         elif mock_broadcast_recv.call_count == 2:
    #             encryption = config["train_info"]["train_params"]["encryption"]
    #             if "paillier" in encryption:
    #                 encryption = encryption["paillier"]
    #                 private_context = Paillier.context(
    #                     encryption["key_bit_size"], encryption["djn_on"])
    #                 return private_context.to_public().serialize()
    #             else:
    #                 return None

    #     mock_broadcast_recv = mocker.patch.object(
    #         BroadcastChannel, "recv", side_effect=mock_func
    #     )
        
    #     mocker.patch.object(
    #         service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
    #     )
    #     mocker.patch.object(
    #         service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
    #     )

    #     xgb_trainer = VerticalXgboostTrainer(conf)
    #     sampled_features, feature_id_mapping = xgb_trainer.col_sample()
    #     cat_columns_after_sampling = list(filter(
    #         lambda x: feature_id_mapping[x] in xgb_trainer.cat_columns, list(feature_id_mapping.keys())))
    #     split_points_after_sampling = [
    #         xgb_trainer.split_points[feature_id_mapping[k]] for k in feature_id_mapping.keys()]

    #     def mock_grad_hess(*args, **kwargs):
    #         private_context = Paillier.context(xgb_trainer.xgb_config.encryption_param.key_bit_size,
    #                                            xgb_trainer.xgb_config.encryption_param.djn_on)
    #         grad = np.random.random(xgb_trainer.xgb_config.num_bins)
    #         hess = np.random.random(xgb_trainer.xgb_config.num_bins)
    #         grad_hess = embed([grad, hess], interval=(1 << 128), precision=64)
    #         enc_grad_hess = Paillier.encrypt(private_context,
    #                                          data=grad_hess,
    #                                          precision=0,  # must be 0
    #                                          obfuscation=True,
    #                                          num_cores=999)
    #         return Paillier.serialize(enc_grad_hess, compression=False)

    #     mocker.patch.object(
    #         BroadcastChannel, "recv", side_effect=mock_grad_hess
    #     )

    #     decision_tree = VerticalDecisionTreeTrainer(tree_param=xgb_trainer.xgb_config,
    #                                                 features=sampled_features,
    #                                                 cat_columns=cat_columns_after_sampling,
    #                                                 split_points=split_points_after_sampling,
    #                                                 channels=xgb_trainer.channels,
    #                                                 encryption_context=xgb_trainer.public_context,
    #                                                 feature_id_mapping=feature_id_mapping,
    #                                                 tree_index=0)

    #     def mock_node(*args, **kwargs):
    #         """
    #         mock the node passing to the trainer
    #         Args:
    #                 *args:
    #                 **kwargs:

    #         Returns:
    #                 an empty None
    #         """
    #         if node_mocker.call_count <= 1:
    #             return Node(id="mock_id")
    #         else:
    #             return None

    #     node_mocker = mocker.patch.object(
    #         decision_tree.tree_node_chann, "recv", side_effect=mock_node
    #     )

    #     mocker.patch.object(
    #         decision_tree.min_split_info_chann, "recv", return_value=[0, 1, []]
    #     )

    #     decision_tree.fit()

    # def test_decision_tree_label_trainer(self, mocker, tmp_factory):
    #     with open("python/algorithm/config/vertical_xgboost/label_trainer.json") as f:
    #         conf = json.load(f)
    #         conf["input"]["trainset"][0]["path"] = str(tmp_factory)
    #         conf["input"]["trainset"][0]["name"] = "train_guest.csv"
    #         conf["input"]["valset"][0]["path"] = str(tmp_factory)
    #         conf["input"]["valset"][0]["name"] = "test_guest.csv"
    #         conf["output"]["path"] = str(tmp_factory)
    #         del conf["input"]["testset"]

    #     mocker.patch.object(
    #         BroadcastChannel, "__init__", return_value=None
    #     )

    #     mocker.patch.object(
    #         BroadcastChannel, "broadcast", return_value=None
    #     )

    #     xgb_label_trainer = VerticalXgboostLabelTrainer(conf)
    #     train_y_pred = np.zeros_like(xgb_label_trainer.train_label) + 0.5
    #     sampled_features, feature_id_mapping = xgb_label_trainer.col_sample()
    #     cat_columns_after_sampling = list(filter(
    #         lambda x: feature_id_mapping[x] in xgb_label_trainer.cat_columns, list(feature_id_mapping.keys())))
    #     split_points_after_sampling = [
    #         xgb_label_trainer.split_points[feature_id_mapping[k]] for k in feature_id_mapping.keys()]
    #     decision_tree = VerticalDecisionTreeLabelTrainer(tree_param=xgb_label_trainer.xgb_config,
    #                                                      y=xgb_label_trainer.train_label,
    #                                                      y_pred=train_y_pred,
    #                                                      features=sampled_features,
    #                                                      cat_columns=cat_columns_after_sampling,
    #                                                      split_points=split_points_after_sampling,
    #                                                      channels=xgb_label_trainer.channels,
    #                                                      encryption_context=xgb_label_trainer.private_context,
    #                                                      feature_id_mapping=feature_id_mapping,
    #                                                      tree_index=0)
    #     mocker_grad_hess = mocker.patch.object(
    #         DualChannel, "__init__", return_value=None
    #     )
    #     decision_tree.summed_grad_hess_channs = {
    #         "node-2": DualChannel(name="summed_grad_hess_node-2")}

    #     def mock_grad_hess(*args, **kwargs):
    #         grad = np.random.random(xgb_label_trainer.xgb_config.num_bins)
    #         hess = np.random.random(xgb_label_trainer.xgb_config.num_bins)
    #         grad_hess = embed([grad, hess], interval=(1 << 128), precision=64)
    #         grad_hess_enc = Paillier.encrypt(xgb_label_trainer.private_context,
    #                                          data=grad_hess,
    #                                          precision=0,  # must be 0
    #                                          obfuscation=True,
    #                                          num_cores=999)

    #         grad_hess_hist_list = []
    #         remote_cat_index = []
    #         grad_hess_hist_list.append(
    #             (grad_hess_enc, [xgb_label_trainer.xgb_config.num_bins]))
    #         if mocker_grad_hess.call_count > 1:
    #             return False, grad_hess_hist_list, remote_cat_index
    #         else:
    #             return True, grad_hess_hist_list, remote_cat_index

    #     mocker_grad_hess = mocker.patch.object(
    #         decision_tree.summed_grad_hess_channs["node-2"], "recv", side_effect=mock_grad_hess
    #     )
    #     decision_tree.fit()

    def test_trainer(self, mocker, tmp_factory):
        # load default config
        with open("python/algorithm/config/vertical_xgboost/trainer.json") as f:
            conf = json.load(f)
            conf["input"]["trainset"][0]["path"] = str(tmp_factory)
            conf["input"]["trainset"][0]["name"] = "train_host.csv"
            conf["input"]["valset"][0]["path"] = str(tmp_factory)
            conf["input"]["valset"][0]["name"] = "test_host.csv"
            conf["output"]["path"] = str(tmp_factory)
            # if conf["train_info"]["train_params"]["downsampling"]["row"]["run_goss"]:
            #     conf["train_info"]["train_params"]["downsampling"]["row"]["top_rate"] = 0.5
            #     conf["train_info"]["train_params"]["downsampling"]["row"]["other_rate"] = 0.5
            del conf["input"]["testset"]

        # mocker channels in VerticalXgboostTrainer.__init__
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=None
        )

        def mock_func(*args, **kwargs):
            """
            mock encryption keys
            Args:
                    *args:
                    **kwargs:

            Returns:
                    the paillier context
            """
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
                        "num_trees": 10,
                        "num_bins": 16,
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
            if mock_broadcast_recv.call_count == 1:
                return config
            elif mock_broadcast_recv.call_count == 2:
                encryption = config["train_info"]["train_params"]["encryption"]
                if "paillier" in encryption:
                    encryption = encryption["paillier"]
                    private_context = Paillier.context(
                        encryption["key_bit_size"], encryption["djn_on"])
                    return private_context.to_public().serialize()
                else:
                    return None

        mocker.patch.object(
            BroadcastChannel, "__init__", return_value=None
        )

        mock_broadcast_recv = mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_func
        )

        xgb_trainer = VerticalXgboostTrainer(conf)

        # mock for iters
        private_context = Paillier.context(2048, True)
        public_context = private_context.to_public()
        xgb_trainer.public_context = public_context

        def mock_grad_hess(*args, **kwargs):
            """
            mock the grad and hess calculation in the label trainer.
            Args:
                    *args:
                    **kwargs:

            Returns:
                    paillier encrypted grad and hess vec
            """
            y = np.array([0, 1] * 60)
            y_pred = np.array([0.5] * 120)
            loss_inst = get_xgb_loss_inst("BCEWithLogitsLoss")
            grad = loss_inst.cal_grad(y, y_pred, after_prediction=True)
            hess = loss_inst.cal_hess(y, y_pred, after_prediction=True)
            grad_hess = embed([grad, hess], interval=(1 << 128), precision=64)
            enc_grad_hess = Paillier.encrypt(context=private_context,
                                             data=grad_hess,
                                             precision=0,  # must be 0
                                             obfuscation=True,
                                             num_cores=999)
            return Paillier.serialize(enc_grad_hess, compression=False)

        def mock_node(*args, **kwargs):
            """
            mock the node passing to the trainer
            Args:
                    *args:
                    **kwargs:

            Returns:
                    an empty None
            """
            if node_mocker.call_count <= 1:
                return Node(id="mock_id")
            else:
                return None

        # mock results from the label trainer according to difference channels
        mocker.patch.object(
            xgb_trainer.channels["individual_grad_hess"], "recv", side_effect=mock_grad_hess
        )
        node_mocker = mocker.patch.object(
            xgb_trainer.channels["tree_node"], "recv", side_effect=mock_node
        )
        mocker.patch.object(
            xgb_trainer.channels["min_split_info"], "recv", return_value=[-1, -1, -1]
        )
        mocker.patch.object(
            xgb_trainer.channels["restart_com"], "recv", return_value=0
        )
        mocker.patch.object(
            xgb_trainer.channels["early_stop_com"], "recv", return_value=False
        )

        xgb_trainer.fit()
        self.check_trainer_output(tmp_factory)

    def test_label_trainer(self, mocker, tmp_factory):
        with open("python/algorithm/config/vertical_xgboost/label_trainer.json") as f:
            conf = json.load(f)
            conf["input"]["trainset"][0]["path"] = str(tmp_factory)
            conf["input"]["trainset"][0]["name"] = "train_guest.csv"
            conf["input"]["valset"][0]["path"] = str(tmp_factory)
            conf["input"]["valset"][0]["name"] = "test_guest.csv"
            conf["output"]["path"] = str(tmp_factory)
            del conf["input"]["testset"]

        mocker.patch.object(
            BroadcastChannel, "__init__", return_value=None
        )

        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=None
        )

        xgb_label_trainer = VerticalXgboostLabelTrainer(conf)
        mocker.patch.object(
            xgb_label_trainer.channels["check_dataset_com"], "collect", return_value=[]
        )
        xgb_label_trainer.fit()
        self.check_label_trainer_output(tmp_factory)

        # cover dual channel created in: VerticalXgboostLabelTrainer.__init__
        mocker.patch.object(
            FedConfig, "get_trainer", return_value=["node_id"]
        )
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        VerticalXgboostLabelTrainer(conf)

    @staticmethod
    def check_label_trainer_output(tmp_factory):
        # 检查是否正确输出了预测值文件
        assert os.path.exists(tmp_factory.join(
            "xgb_prediction_train_[STAGE_ID].csv"))
        assert os.path.exists(tmp_factory.join(
            "xgb_prediction_val_[STAGE_ID].csv"))

        # 检查是否正确输出了模型文件
        assert os.path.exists(tmp_factory.join(
            "vertical_xgboost_[STAGE_ID].model"))

        # 检查是否正确输出了model config
        assert os.path.exists(tmp_factory.join("model_config.json"))
        with open(tmp_factory.join("model_config.json")) as f:
            model_config = json.load(f)
        assert model_config[0]["class_name"] == "VerticalXGBooster"
        assert model_config[0]["filename"] == "vertical_xgboost_[STAGE_ID].model"

        # 检查是否正确输出了feature importance文件
        assert os.path.exists(tmp_factory.join(
            "xgb_feature_importance_[STAGE_ID].csv"))

    @staticmethod
    def check_trainer_output(tmp_factory):
        # 检查是否正确输出了模型文件
        assert os.path.exists(tmp_factory.join(
            "vertical_xgboost_[STAGE_ID].model"))

        # 检查是否正确输出了model config
        assert os.path.exists(tmp_factory.join("model_config.json"))
        with open(tmp_factory.join("model_config.json")) as f:
            model_config = json.load(f)
        assert model_config[0]["class_name"] == "VerticalXGBooster"
        assert model_config[0]["filename"] == "vertical_xgboost_[STAGE_ID].model"

    def test_predict_label_trainer(self, get_label_trainer_infer_conf, mocker, tmp_factory):
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            ApplyResult, "get", return_value={"0_4lN0P7QTwWq25Eei": np.array([1] * 50 + [0] * 30),
                                              "0_gw94EBW5tiD8kCqG": np.array([1] * 25 + [0] * 55),
                                              "0_vpKZWumTxYcojXLq": np.array([1] * 75 + [0] * 5)}
        )
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "collect", return_value=[{"test": (80, 2)}]
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        xgb_label_trainer = VerticalXgboostLabelTrainer(
            get_label_trainer_infer_conf)
        xgb_label_trainer.predict()
        df = pd.read_csv(tmp_factory.join("predicted_probabilities_train.csv"))
        assert (df["pred"] > 0.5).sum() == 50

    def test_predict_empty_testset(self, get_label_trainer_infer_conf, mocker, tmp_factory):
        conf = copy.deepcopy(get_label_trainer_infer_conf)
        del conf["input"]["testset"]
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            ApplyResult, "get", return_value={"0_4lN0P7QTwWq25Eei": np.array([1] * 50 + [0] * 30),
                                              "0_gw94EBW5tiD8kCqG": np.array([1] * 25 + [0] * 55),
                                              "0_vpKZWumTxYcojXLq": np.array([1] * 75 + [0] * 5)}
        )
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=None
        )
        mocker.patch.object(
            BroadcastChannel, "collect", return_value=[{"test": (80, 2)}]
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        xgb_label_trainer = VerticalXgboostLabelTrainer(conf)
        xgb_label_trainer.predict()
        df = pd.read_csv(tmp_factory.join("predicted_probabilities_train.csv"))
        assert df.shape == (80, 2)

    def test_predict_trainer(self, get_trainer_infer_conf, mocker, tmp_factory):
        mocker.patch.object(
            DualChannel, "__init__", return_value=None
        )
        mocker.patch.object(
            DualChannel, "send", return_value=0
        )
        mocker.patch.object(
            BroadcastChannel, "send", return_value=0
        )
        
        def mock_func(*args, **kwargs):
            config = {
                "train_info": {
                    "train_params": {
                        "lossfunc": {
                            "BCEWithLogitsLoss": {}
                        },
                        "batch_size_val": 40960
                    }
                }
            }
            return config

        mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_func
        )
        
        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )
        
        xgb_trainer = VerticalXgboostTrainer(get_trainer_infer_conf)
        xgb_trainer.predict()
