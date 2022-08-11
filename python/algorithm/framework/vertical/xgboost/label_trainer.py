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
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from algorithm.core.activation import sigmoid
from algorithm.core.data_io import ValidationNumpyDataset
from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.tree_structure import Tree
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.evaluation.metrics import ThresholdCutter
from common.utils.algo_utils import earlyStopping
from common.utils.logger import logger
from common.utils.tree_transfer import label_trainer_tree_transfer
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from .base import VerticalXgboostBase
from .decision_tree_label_trainer import VerticalDecisionTreeLabelTrainer


class VerticalXgboostLabelTrainer(VerticalXgboostBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, is_label_trainer=True, *args, **kwargs)

        self.channels = {}
        self.channels["encryption_context"] = BroadcastChannel(name="encryption_context")
        self.channels["individual_grad_hess"] = BroadcastChannel(name="individual_grad_hess")
        self.channels["tree_node"] = BroadcastChannel(name="tree_node")

        summed_grad_hess_channs: Dict[str, DualChannel] = {}
        min_split_info_channs: Dict[str, DualChannel] = {}
        sample_index_after_split_channs: Dict[str, DualChannel] = {}
        val_com: Dict[str, DualChannel] = {}
        early_stop_com: Dict[str, DualChannel] = {}

        for party_id in FedConfig.get_trainer():
            summed_grad_hess_channs[party_id] = \
                DualChannel(name="summed_grad_hess_" + party_id, ids=[FedConfig.node_id, party_id])

            min_split_info_channs[party_id] = \
                DualChannel(name="min_split_info_" + party_id, ids=[FedConfig.node_id, party_id])

            sample_index_after_split_channs[party_id] = \
                DualChannel(name="sample_index_after_split_" + party_id, ids=[FedConfig.node_id, party_id])

            val_com[party_id] = \
                DualChannel(name="val_com_" + party_id, ids=[FedConfig.node_id, party_id])

            early_stop_com[party_id] = \
                DualChannel(name="early_stop_com_" + party_id, ids=[FedConfig.node_id, party_id])

        self.channels["summed_grad_hess"] = summed_grad_hess_channs
        self.channels["min_split_info"] = min_split_info_channs
        self.channels["sample_index_after_split"] = sample_index_after_split_channs
        self.channels["val_com"] = val_com
        self.channels["early_stop_com"] = early_stop_com

        if isinstance(self.xgb_config.encryption_param, (PlainParam, type(None))):
            self.private_context = None
        elif isinstance(self.xgb_config.encryption_param, PaillierParam):
            self.private_context = Paillier.context(self.xgb_config.encryption_param.key_bit_size,
                                                    self.xgb_config.encryption_param.djn_on)
            self.public_context = self.private_context.to_public()
            self.channels["encryption_context"].broadcast(self.public_context.serialize(), use_pickle=False)
        else:
            raise TypeError(f"Encryption param type {type(self.xgb_config.encryption_param)} not valid.")

        self.es = earlyStopping(key=self.xgb_config.early_stopping_param["key"],
                                patience=self.xgb_config.early_stopping_param["patience"],
                                delta=self.xgb_config.early_stopping_param["delta"])
        self.best_round = -1
        self.best_prediction_val = None
        self.best_prediction_train = None
        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.identity,
            "filename": self.output.get("model", {"name": "vertical_xgboost_guest.pt"})["name"]
        }]

    def fit(self):
        train_y_pred_primitive, tree_list = np.zeros_like(self.train_label), []
        val_y_pred_primitive = np.zeros_like(self.val_label)

        loss_inst = get_xgb_loss_inst(self.xgb_config.loss_param['method'])
        train_y_pred, val_y_pred = loss_inst.predict(train_y_pred_primitive), loss_inst.predict(val_y_pred_primitive)

        for tree_idx in range(self.xgb_config.num_trees):
            logger.info("Tree {} start training.".format(tree_idx))
            # train section
            sampled_features, feature_id_mapping = self.col_sample()
            trainer = VerticalDecisionTreeLabelTrainer(tree_param=self.xgb_config,
                                                       y=self.train_label,
                                                       y_pred=train_y_pred,
                                                       features=sampled_features,
                                                       split_points=self.split_points,
                                                       channels=self.channels,
                                                       encryption_context=self.private_context,
                                                       feature_id_mapping=feature_id_mapping,
                                                       tree_index=tree_idx)
            tree = trainer.fit()
            for party_id in FedConfig.get_trainer():
                self.channels["early_stop_com"][party_id].send(tree.root_node.is_leaf)
            if tree.root_node.is_leaf:
                logger.warning("Tree {} root is leaf, mission stopped.".format(tree_idx))
                break

            self.update_feature_importance(trainer.get_feature_importance())

            if self.xgb_config.run_goss:
                train_y_pred_primitive = self.validation(self.train_dataset, tree, train_y_pred_primitive)
            else:
                for node_id, node in tree.nodes.items():
                    if node.is_leaf:
                        train_y_pred_primitive[node.sample_index] += node.weight * self.xgb_config.learning_rate

            train_y_pred = loss_inst.predict(train_y_pred_primitive)
            if self.interaction_params.get("echo_training_metrics"):
                train_loss = loss_inst.cal_loss(self.train_label, train_y_pred_primitive, after_prediction=False)
                self._calc_metrics(self.train_label, train_y_pred, tree_idx, stage="training",
                                   loss={loss_inst.name: train_loss})

            tree.clear_training_info()

            tree_list.append(label_trainer_tree_transfer(tree=tree))
            logger.info("Tree {} training done.".format(tree_idx))

            # validation section
            logger.info("Validation on tree {} start.".format(tree_idx))
            val_y_pred_primitive = self.validation(self.val_dataset, tree, val_y_pred_primitive)
            val_y_pred = loss_inst.predict(val_y_pred_primitive)
            val_loss = loss_inst.cal_loss(self.val_label, val_y_pred_primitive, after_prediction=False)
            metric = self._calc_metrics(self.val_label, val_y_pred, tree_idx, stage="validation",
                                        loss={loss_inst.name: val_loss})
            logger.info("Validation on tree {} done.".format(tree_idx))
            if self.xgb_config.early_stopping_param["patience"] > 0:
                early_stop_flag, save_flag = self.es(metric)
            else:
                early_stop_flag, save_flag = False, True

            if save_flag:
                self.best_round = tree_idx + 1
                self.best_prediction_train = copy.deepcopy(train_y_pred)
                self.best_prediction_val = copy.deepcopy(val_y_pred)

            for party_id in FedConfig.get_trainer():
                self.channels["early_stop_com"][party_id].send(early_stop_flag)
            if early_stop_flag:
                logger.info("label trainer early stopped. best round: {}.".format(self.best_round))
                break

            if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get(
                    "save_frequency") == 0:
                self.save(tree_list, epoch=tree_idx + 1)
                self._write_prediction(self.train_label, train_y_pred, self.train_ids, epoch=tree_idx + 1)
                self._write_prediction(self.val_label, val_y_pred, self.val_ids, epoch=tree_idx + 1, stage='val')

        # model preserve
        if self.xgb_config.early_stopping_param["patience"] <= 0:
            self.best_round = len(tree_list)
            self.best_prediction_train = copy.deepcopy(train_y_pred)
            self.best_prediction_val = copy.deepcopy(val_y_pred)
        logger.info("num trees: %d, best: %d" % (len(tree_list), self.best_round))

        if tree_list:
            self.save(tree_list, final=True)
            self._write_prediction(self.train_label, train_y_pred, self.train_ids, final=True)
            self._write_prediction(self.val_label, val_y_pred, self.val_ids, final=True, stage='val')
        else:
            logger.error("Model is none, ture off goss and subsample please.")
            raise SystemError("Model is none, ture off goss and subsample please.")

    def save(self, tree_list, epoch: int = None, final: bool = False):
        if final:
            save_model_config(stage_model_config=self.export_conf, save_path=Path(self.output.get("model")["path"]))

        save_dir = str(Path(self.output.get("model")["path"]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # dump out ks plot
        suggest_threshold = 0.5
        if "ks" in self.xgb_config.metrics or "auc_ks" in self.xgb_config.metrics:
            tc = ThresholdCutter(os.path.join(save_dir, "ks_plot_valid.csv"))
            tc.cut_by_value(self.val_label, self.best_prediction_val)
            suggest_threshold = tc.bst_threshold
            tc.save()
            if self.interaction_params.get("echo_training_metrics"):
                tc = ThresholdCutter(os.path.join(save_dir, "ks_plot_train.csv"))
                tc.cut_by_value(self.train_label, self.best_prediction_train)
                tc.save()

        model_name_list = self.output.get("model")["name"].split(".")
        name_prefix, name_postfix = ".".join(model_name_list[:-1]), model_name_list[-1]
        if not final and epoch:
            model_name = name_prefix + "_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix
        model_path = os.path.join(save_dir, model_name)

        node_id_of_owner = {}
        for tree in tree_list:
            for node_id in tree.nodes:
                owner_id = tree.nodes[node_id].owner_id
                if owner_id is None:
                    continue
                if owner_id not in node_id_of_owner:
                    node_id_of_owner[owner_id] = [node_id]
                else:
                    node_id_of_owner[owner_id].append(node_id)

        for owner_id in node_id_of_owner:
            node_id_of_owner[owner_id].sort()

        node_id_group = {}
        for k, v in node_id_of_owner.items():
            node_id_group[v[0]] = v

        xgb_output = {
            "trees": tree_list[:self.best_round],
            "num_trees": self.best_round,
            "lr": self.xgb_config.learning_rate,
            "max_depth": self.xgb_config.max_depth,
            "suggest_threshold": suggest_threshold,
            "node_id_group": node_id_group
        }
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_output, f)
        logger.info("model saved as: {}.".format(model_path))

        self.make_readable_feature_importance(os.path.join(save_dir, "feature_importances.csv"))

    def make_readable_feature_importance(self, file_name):
        with open(file_name, "w") as f:
            f.write("owner_id,fid,importance\n")
            normalizer = np.sum([_.get() for _ in self.feature_importances_.values()])
            for k, v in sorted(self.feature_importances_.items(), key=lambda d: d[1], reverse=True):
                f.write("%s,%s,%.6g\n" % (k[0], k[1], v.get() / normalizer))

    def validation(self, data: ValidationNumpyDataset, tree: Tree, y_pred: np.ndarray) -> np.ndarray:
        """ Function used for validation section.

        Args:
            data: validation dataset.
            tree: tree fitted in this iteration.
            y_pred: prediction.

        Returns: y_pred for validation.

        """
        for idx, (x, y) in enumerate(data):
            p = y_pred[(idx * self.bs): (idx * self.bs) + x.shape[0]]
            node_info, node_sample = {}, {}
            for k, channel in self.channels["val_com"].items():
                node_info.update(channel.recv())
            for depth in range(self.xgb_config.max_depth + 1):
                nodes = tree.search_nodes(depth)
                for node in nodes:
                    if node.is_leaf:
                        sample_index = node_sample[node.parent_node_id][node.linkage]
                        p[sample_index] += node.weight * self.xgb_config.learning_rate
                    else:
                        if node.id not in node_info:
                            node_info[node.id] = x[:, node.split_info.feature_idx] < node.split_info.split_point
                        if node.parent_node_id is not None:
                            node_sample[node.id] = {
                                "left": np.intersect1d(node_sample[node.parent_node_id][node.linkage],
                                                       np.argwhere(node_info[node.id]).ravel()),
                                "right": np.intersect1d(node_sample[node.parent_node_id][node.linkage],
                                                        np.argwhere(~node_info[node.id]).ravel())
                            }
                        else:
                            node_sample[node.id] = {
                                "left": np.argwhere(node_info[node.id]).ravel(),
                                "right": np.argwhere(~node_info[node.id]).ravel()
                            }
            y_pred[(idx * self.bs): (idx * self.bs + x.shape[0])] = p
        return y_pred

    def update_feature_importance(self, tree_feature_importance):
        for (owner_id, fid) in tree_feature_importance:
            if (owner_id, fid) not in self.feature_importances_:
                self.feature_importances_[(owner_id, fid)] = tree_feature_importance[(owner_id, fid)]
            else:
                self.feature_importances_[(owner_id, fid)] += tree_feature_importance[(owner_id, fid)]
        logger.debug("cur feature importance {}".format(self.feature_importances_))

    def col_sample(self) -> tuple[Any, dict]:
        col_size = self.train_features.shape[1]
        if 0 < self.xgb_config.subsample_feature_rate <= 1:
            sample_num = int(col_size * self.xgb_config.subsample_feature_rate)
        else:
            sample_num = col_size
        sampled_idx = np.sort(np.random.choice(col_size, sample_num, replace=False))
        feature_id_mapping = {a: b for a, b in enumerate(sampled_idx)}
        sampled_features = self.train_features.iloc[:, sampled_idx]
        return sampled_features, feature_id_mapping

    def load_model(self):
        model_path = Path(
            self.input.get("pretrain_model", {}).get("path", ''),
            self.input.get("pretrain_model", {}).get("name", '')
        )
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.xgb_config.learning_rate = model["lr"]
        self.xgb_config.max_depth = model["max_depth"]
        trees = []
        for tree in model["trees"]:
            t = Tree(tree.party_id, tree.root_node_id)
            t.nodes = tree.nodes
            t.root_node = tree.root_node
            trees.append(t)
        return trees

    def predict(self):
        trees = self.load_model()
        test_y_pred_primitive = np.zeros_like(self.test_label)
        for tree in trees:
            test_y_pred_primitive = self.validation(self.test_dataset, tree, test_y_pred_primitive)
        test_y_pred = sigmoid(test_y_pred_primitive)
        save_path = self.output.get("testset", {}).get("path", '')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = Path(save_path, self.output.get("testset", {}).get("name", ''))
        logger.info("predicted results saved at {}".format(save_path))
        pd.DataFrame({"id": self.test_ids, "pred": test_y_pred}).to_csv(
            save_path, float_format="%.6g", index=False, header=True
        )
