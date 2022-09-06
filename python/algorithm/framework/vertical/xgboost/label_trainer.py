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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pathos.pools import ThreadPool

from algorithm.core.data_io import NdarrayIterator
from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.tree_structure import BoostingTree, Tree
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.evaluation.metrics import ThresholdCutter
from common.utils.algo_utils import earlyStopping
from common.utils.logger import logger
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from .base import VerticalXgboostBase
from .decision_tree_label_trainer import VerticalDecisionTreeLabelTrainer


class VerticalXgboostLabelTrainer(VerticalXgboostBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, is_label_trainer=True, *args, **kwargs)
        self.party_id = FedConfig.node_id

        self.channels = dict()
        self.channels["encryption_context"] = BroadcastChannel(
            name="encryption_context")
        self.channels["individual_grad_hess"] = BroadcastChannel(
            name="individual_grad_hess")
        self.channels["tree_node"] = BroadcastChannel(name="tree_node")

        summed_grad_hess_channs: Dict[str, DualChannel] = {}
        min_split_info_channs: Dict[str, DualChannel] = {}
        sample_index_after_split_channs: Dict[str, DualChannel] = {}
        val_com: Dict[str, DualChannel] = {}
        restart_com: Dict[str, DualChannel] = {}
        early_stop_com: Dict[str, DualChannel] = {}

        for party_id in FedConfig.get_trainer():
            summed_grad_hess_channs[party_id] = \
                DualChannel(name="summed_grad_hess_" + party_id,
                            ids=[FedConfig.node_id, party_id])

            min_split_info_channs[party_id] = \
                DualChannel(name="min_split_info_" + party_id,
                            ids=[FedConfig.node_id, party_id])

            sample_index_after_split_channs[party_id] = \
                DualChannel(name="sample_index_after_split_" +
                            party_id, ids=[FedConfig.node_id, party_id])

            val_com[party_id] = \
                DualChannel(name="val_com_" + party_id,
                            ids=[FedConfig.node_id, party_id])

            restart_com[party_id] = \
                DualChannel(name="restart_com_" + party_id,
                            ids=[FedConfig.node_id, party_id])

            early_stop_com[party_id] = \
                DualChannel(name="early_stop_com_" + party_id,
                            ids=[FedConfig.node_id, party_id])

        self.channels["summed_grad_hess"] = summed_grad_hess_channs
        self.channels["min_split_info"] = min_split_info_channs
        self.channels["sample_index_after_split"] = sample_index_after_split_channs
        self.channels["val_com"] = val_com
        self.channels["restart_com"] = restart_com
        self.channels["early_stop_com"] = early_stop_com

        if isinstance(self.xgb_config.encryption_param, (PlainParam, type(None))):
            self.private_context = None
        elif isinstance(self.xgb_config.encryption_param, PaillierParam):
            self.private_context = Paillier.context(self.xgb_config.encryption_param.key_bit_size,
                                                    self.xgb_config.encryption_param.djn_on)
            self.public_context = self.private_context.to_public()
            self.channels["encryption_context"].broadcast(
                self.public_context.serialize(), use_pickle=False)
        else:
            raise TypeError(
                f"Encryption param type {type(self.xgb_config.encryption_param)} not valid.")

        self.es = earlyStopping(key=self.xgb_config.early_stopping_param["key"],
                                patience=self.xgb_config.early_stopping_param["patience"],
                                delta=self.xgb_config.early_stopping_param["delta"])
        self.best_round = -1
        self.best_prediction_val = None
        self.best_prediction_train = None
        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.identity,
            "filename": self.output.get("model", {"name": "vertical_xgboost_guest.json"})["name"]
        }]

    def fit(self):
        boosting_tree = BoostingTree()

        # train_y_pred_primitive, tree_list = np.zeros_like(self.train_label), []
        train_y_pred_primitive = np.zeros_like(self.train_label)
        val_y_pred_primitive = np.zeros_like(self.val_label)

        loss_inst = get_xgb_loss_inst(self.xgb_config.loss_param['method'])
        train_y_pred, val_y_pred = loss_inst.predict(
            train_y_pred_primitive), loss_inst.predict(val_y_pred_primitive)

        for tree_idx in range(self.xgb_config.num_trees):
            logger.info("Tree {} start training.".format(tree_idx))

            # 0: no need to restart, 1: restart, 2: max number of try reached
            restart_status = 1
            while True:
                # train section
                sampled_features, feature_id_mapping = self.col_sample()
                cat_columns_after_sampling = list(filter(
                    lambda x: feature_id_mapping[x] in self.cat_columns, list(feature_id_mapping.keys())))
                split_points_after_sampling = [
                    self.split_points[feature_id_mapping[k]] for k in feature_id_mapping.keys()]

                trainer = VerticalDecisionTreeLabelTrainer(tree_param=self.xgb_config,
                                                           y=self.train_label,
                                                           y_pred=train_y_pred,
                                                           features=sampled_features,
                                                           cat_columns=cat_columns_after_sampling,
                                                           split_points=split_points_after_sampling,
                                                           channels=self.channels,
                                                           encryption_context=self.private_context,
                                                           feature_id_mapping=feature_id_mapping,
                                                           tree_index=tree_idx)
                tree = trainer.fit()

                if not tree.root_node.is_leaf:
                    restart_status = 0
                else:
                    if self.xgb_config.early_stopping_param["patience"] <= 0:
                        # if not set patience, terminate immediately
                        restart_status = 2
                    else:
                        self.es.counter += 1

                if self.es.counter == self.xgb_config.early_stopping_param["patience"]:
                    restart_status = 2

                for party_id in FedConfig.get_trainer():
                    self.channels["restart_com"][party_id].send(restart_status)

                if restart_status != 1:
                    break

                logger.info(f"label trainer tree {tree_idx} training restart.")

            if restart_status == 2:
                logger.info("label trainer early stopped because a tree's root is leaf, best round: {}.".format(
                    self.best_round))
                break

            self.update_feature_importance(trainer.get_feature_importance())

            if self.xgb_config.run_goss:
                train_y_pred_primitive += self.predict_on_tree(
                    tree, self.train_dataset) * self.xgb_config.learning_rate
            else:
                for _, node in tree.nodes.items():
                    if node.is_leaf:
                        train_y_pred_primitive[node.sample_index] += node.weight * \
                            self.xgb_config.learning_rate

            train_y_pred = loss_inst.predict(train_y_pred_primitive)
            if self.interaction_params.get("echo_training_metrics"):
                train_loss = loss_inst.cal_loss(
                    self.train_label, train_y_pred_primitive, after_prediction=False)
                self._calc_metrics(self.train_label, train_y_pred, tree_idx, stage="training", loss={
                                   loss_inst.name: train_loss})

            tree.clear_training_info()

            boosting_tree.append(tree=tree,
                                 lr=self.xgb_config.learning_rate,
                                 max_depth=self.xgb_config.max_depth)
            logger.info("Tree {} training done.".format(tree_idx))

            # validation section
            logger.info("Validation on tree {} start.".format(tree_idx))
            val_y_pred_primitive += self.predict_on_tree(
                tree, self.val_dataset) * self.xgb_config.learning_rate
            val_y_pred = loss_inst.predict(val_y_pred_primitive)
            val_loss = loss_inst.cal_loss(
                self.val_label, val_y_pred_primitive, after_prediction=False)
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
                logger.info(
                    "label trainer early stopped. best round: {}.".format(self.best_round))
                break

            if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get("save_frequency") == 0:
                self.save(boosting_tree, epoch=tree_idx+1)
                self._write_prediction(
                    self.train_label, train_y_pred, self.train_ids, epoch=tree_idx + 1)
                self._write_prediction(
                    self.val_label, val_y_pred, self.val_ids, epoch=tree_idx + 1, stage='val')

        # model preserve
        if self.xgb_config.early_stopping_param["patience"] <= 0:
            self.best_round = len(boosting_tree)
            self.best_prediction_train = copy.deepcopy(train_y_pred)
            self.best_prediction_val = copy.deepcopy(val_y_pred)
        logger.info("num trees: %d, best: %d" %
                    (len(boosting_tree), self.best_round))

        if boosting_tree.trees:
            self.save(boosting_tree, final=True)
            self._write_prediction(
                self.train_label, train_y_pred, self.train_ids, final=True)
            self._write_prediction(
                self.val_label, val_y_pred, self.val_ids, final=True, stage='val')
        else:
            logger.error("Model is none, ture off goss and subsample please.")
            raise SystemError(
                "Model is none, ture off goss and subsample please.")

    def save(self, boosting_tree: BoostingTree, epoch: Optional[int] = None, final: bool = False):
        if final:
            save_model_config(stage_model_config=self.export_conf, save_path=Path(
                self.output.get("model")["path"]))

        save_dir = str(Path(self.output.get("model")["path"]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # dump out ks plot
        suggest_threshold = 0.5
        if "ks" in self.xgb_config.metrics or "auc_ks" in self.xgb_config.metrics:
            tc = ThresholdCutter(os.path.join(save_dir, "ks_plot_valid.csv"))
            tc.cut_by_value(self.val_label, self.best_prediction_val)
            suggest_threshold = float(tc.bst_threshold)
            tc.save()
            if self.interaction_params.get("echo_training_metrics"):
                tc = ThresholdCutter(os.path.join(
                    save_dir, "ks_plot_train.csv"))
                tc.cut_by_value(self.train_label, self.best_prediction_train)
                tc.save()

        model_name_list = self.output.get("model")["name"].split(".")
        name_prefix, name_postfix = ".".join(
            model_name_list[:-1]), model_name_list[-1]
        if not final and epoch:
            model_name = name_prefix + "_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix
        model_path = os.path.join(save_dir, model_name)

        xgb_output = boosting_tree[:self.best_round].to_dict(
            suggest_threshold, compute_group=True)

        with open(model_path, 'w') as f:
            json.dump(xgb_output, f)
        logger.info("model saved as: {}.".format(model_path))

        self.make_readable_feature_importance(
            os.path.join(save_dir, "feature_importances.csv"))

    def make_readable_feature_importance(self, file_name):
        with open(file_name, "w") as f:
            f.write("owner_id,fid,importance\n")
            normalizer = np.sum([_.get()
                                for _ in self.feature_importances_.values()])
            for k, v in sorted(self.feature_importances_.items(), key=lambda d: d[1], reverse=True):
                f.write("%s,%s,%.6g\n" % (k[0], k[1], v.get() / normalizer))

    def _make_indicator_for_prediction(self, tree: Tree, feature: np.ndarray):
        indicator = {}
        for node_id, node in tree.nodes.items():
            if not node.is_leaf and node.split_info.owner_id == self.party_id:
                feature_idx = node.split_info.feature_idx
                data = feature[:, feature_idx]
                if node.split_info.is_category:
                    indicator[node_id] = np.isin(
                        data, node.split_info.left_cat)
                else:
                    indicator[node_id] = (data < node.split_info.split_point)
        return indicator

    def _gen_prediction(self, tree: Tree, indicator: Dict[str, np.ndarray], feature: np.ndarray):
        prediction = np.zeros((len(feature),), dtype=np.float32)
        depth = 0
        sample_in_node = {}

        while True:
            node_list = tree.search_nodes(depth)
            if not node_list:
                break

            for node in node_list:
                if node.is_leaf:
                    prediction[sample_in_node[node.id]] = node.weight
                else:
                    if depth == 0:
                        sample_in_node[node.left_node_id] = np.where(
                            indicator[node.id] == 1)[0]
                        sample_in_node[node.right_node_id] = np.where(
                            indicator[node.id] == 0)[0]
                    else:
                        sample_in_node[node.left_node_id] = np.intersect1d(
                            sample_in_node[node.id], np.where(indicator[node.id] == 1)[0])
                        sample_in_node[node.right_node_id] = np.intersect1d(
                            sample_in_node[node.id], np.where(indicator[node.id] == 0)[0])

            depth += 1
        return prediction

    def predict_on_tree(self, tree: Tree, data_iterator: NdarrayIterator) -> np.ndarray:
        prediction = np.zeros((len(data_iterator.data),), dtype=np.float32)
        indicator = {}

        def _update_local(tree, data):
            res = self._make_indicator_for_prediction(tree, data)
            return res

        def _update_remote(channel, len_data):
            remote_indicator = channel.recv()
            res = {k: np.unpackbits(v)[:len_data]
                   for k, v in remote_indicator.items()}
            return res

        thread_pool = ThreadPool(len(self.channels["val_com"]) + 1)

        for i, data in enumerate(data_iterator):
            indicator = {}

            threads = []
            threads.append(thread_pool.apipe(_update_local, tree, data))
            for party_id in self.channels["val_com"]:
                threads.append(thread_pool.apipe(
                    _update_remote, self.channels["val_com"][party_id], len(data)))

            for t in threads:
                indicator.update(t.get())

            prediction[i * data_iterator.bs: (
                i + 1) * data_iterator.bs] = self._gen_prediction(tree, indicator, data)
        return prediction

    # Non-parallelized version
    # def predict_on_tree(self, tree: Tree, data_iterator: NdarrayIterator) -> np.ndarray:
    #     prediction = np.zeros((len(data_iterator.data),), dtype=np.float32)

    #     for i, data in enumerate(data_iterator):
    #         indicator = {}
    #         indicator.update(self._make_indicator_for_prediction(tree, data))

    #         for party_id in self.channels["val_com"]:
    #             remote_indicator = self.channels["val_com"][party_id].recv()
    #             indicator.update({k: np.unpackbits(v)[:len(data)] for k, v in remote_indicator.items()})

    #         prediction[i * data_iterator.bs: (i + 1) * data_iterator.bs] = self._gen_prediction(tree, indicator, data)

    #     return prediction

    def predict_on_boosting_tree(self, boosting_tree: BoostingTree, data_iterator: NdarrayIterator) -> np.ndarray:
        prediction = np.zeros((len(data_iterator.data),), dtype=np.float32)

        def _update_local(trees, data):
            res = {}
            for tree in trees:
                res.update(self._make_indicator_for_prediction(tree, data))
            return res

        def _update_remote(channel, len_data):
            remote_indicator = channel.recv()
            res = {k: np.unpackbits(v)[:len_data]
                   for k, v in remote_indicator.items()}
            return res

        thread_pool = ThreadPool(len(self.channels["val_com"]) + 1)

        for i, data in enumerate(data_iterator):
            indicator = {}

            threads = []
            threads.append(thread_pool.apipe(
                _update_local, boosting_tree.trees, data))
            for party_id in self.channels["val_com"]:
                threads.append(thread_pool.apipe(
                    _update_remote, self.channels["val_com"][party_id], len(data)))

            for t in threads:
                indicator.update(t.get())

            for j, tree in enumerate(boosting_tree.trees):
                prediction[i * data_iterator.bs: (i + 1) * data_iterator.bs] += \
                    self._gen_prediction(
                        tree, indicator, data) * boosting_tree.lr[j]

        return prediction

    # Non-parallelized version
    # def predict_on_boosting_tree(self, boosting_tree: BoostingTree, data_iterator: NdarrayIterator) -> np.ndarray:
    #     prediction = np.zeros((len(data_iterator.data),), dtype=np.float32)

    #     for i, data in enumerate(data_iterator):
    #         indicator = {}
    #         for tree in boosting_tree.trees:
    #             indicator.update(self._make_indicator_for_prediction(tree, data))

    #         for party_id in self.channels["val_com"]:
    #             remote_indicator = self.channels["val_com"][party_id].recv()
    #             indicator.update({k: np.unpackbits(v)[:len(data)] for k, v in remote_indicator.items()})

    #         for j, tree in enumerate(boosting_tree.trees):
    #             prediction[i * data_iterator.bs: (i + 1) * data_iterator.bs] += \
    #                 self._gen_prediction(tree, indicator, data) * boosting_tree.lr[j]

    #     return prediction

    def update_feature_importance(self, tree_feature_importance):
        for (owner_id, fid) in tree_feature_importance:
            if (owner_id, fid) not in self.feature_importances_:
                self.feature_importances_[
                    (owner_id, fid)] = tree_feature_importance[(owner_id, fid)]
            else:
                self.feature_importances_[
                    (owner_id, fid)] += tree_feature_importance[(owner_id, fid)]
        logger.debug("cur feature importance {}".format(
            self.feature_importances_))

    def load_model(self):
        model_path = Path(
            self.input.get("pretrain_model", {}).get("path", ''),
            self.input.get("pretrain_model", {}).get("name", '')
        )
        with open(model_path, 'rb') as f:
            json_dict = json.load(f)

        boosting_tree = BoostingTree.from_dict(json_dict)
        return boosting_tree

    def check_dataset(self):
        self.channels["check_dataset_com"] = BroadcastChannel(name="check_dataset_com")
        data_lens = self.channels["check_dataset_com"].collect()
        n = data_lens[0]
        for _ in data_lens:
            if n != _:
                raise ValueError("Lengths of the datasets mismatched.")
        if self.test_dataset:
            assert len(self.test_dataset), "Lengths of the datasets mismatched."
        else:
            self.test_dataset = NdarrayIterator(np.zeros((n, 0)), self.bs)
            self.test_ids = np.arange(n)

    def predict(self):
        self.check_dataset()
        boosting_tree = self.load_model()
        test_y_pred_primitive = self.predict_on_boosting_tree(boosting_tree=boosting_tree,
                                                              data_iterator=self.test_dataset)
        loss_inst = get_xgb_loss_inst(boosting_tree.loss_method)
        test_y_pred = loss_inst.predict(test_y_pred_primitive)
        save_path = self.output.get("testset", {}).get("path", '')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = Path(save_path, self.output.get(
            "testset", {}).get("name", ''))
        logger.info("predicted results saved at {}".format(save_path))
        pd.DataFrame({"id": self.test_ids, "pred": test_y_pred}).to_csv(
            save_path, float_format="%.6g", index=False, header=True
        )
