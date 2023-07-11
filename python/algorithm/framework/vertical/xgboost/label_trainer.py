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
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathos.pools import ThreadPool

from algorithm.core.data_io import NdarrayIterator
from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.tree_structure import BoostingTree, Tree
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
from common.checker.matcher import get_matched_config
from common.checker.x_types import All
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.algo_utils import earlyStopping
from common.utils.logger import logger
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from service.fed_node import FedNode
from .base import VerticalXgboostBase
from .decision_tree_label_trainer import VerticalDecisionTreeLabelTrainer
from service.fed_control import ProgressCalculator
from common.utils.model_io import ModelIO


class VerticalXgboostLabelTrainer(VerticalXgboostBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        self.channels = dict()
        self.channels["sync"] = BroadcastChannel(name="sync")
        self._sync_config(train_conf)
        super().__init__(train_conf, is_label_trainer=True, *args, **kwargs)
        self.party_id = FedConfig.node_id

        self.channels["encryption_context"] = BroadcastChannel(
            name="encryption_context")
        self.channels["individual_grad_hess"] = BroadcastChannel(
            name="individual_grad_hess")
        self.channels["tree_node"] = BroadcastChannel(name="tree_node")
        self.channels["check_dataset_com"] = BroadcastChannel(
            name="check_dataset_com")

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
        if self.train_features is not None:
            input_schema = ','.join([_ for _ in self.train_features.columns if _ not in set(["y", "id"])])
        else:
            input_schema = ""

        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.identity,
            "filename": self.output.get("proto_model", {}).get("name", ''),
            # "filename": self.output.get("proto_model", {"name": "vertical_xgboost_guest.pmodel"})["name"],
            "input_schema": input_schema,
            "version": '1.4.0'
        }]

    def _sync_config(self, config):
        sync_rule = {
            "train_info": {
                "interaction_params": All(),
                "train_params": {
                    "lossfunc": All(),
                    "num_trees": All(),
                    "num_bins": All(),
                    "batch_size_val": All(),
                    "downsampling": {
                        "row": {
                            "run_goss": All()
                        }
                    },
                    "encryption": All()
                }
            }
        }

        config_to_sync = get_matched_config(config, sync_rule)
        self.channels["sync"].broadcast(config_to_sync)

    def fit(self):
        f_names = self.channels["sync"].collect()
        self.remote_f_names = {}
        for name_dict in f_names:
            self.remote_f_names.update(name_dict)
            
        self.check_dataset()
        boosting_tree = BoostingTree()

        # train_y_pred_primitive, tree_list = np.zeros_like(self.train_label), []
        train_y_pred_primitive = np.zeros_like(self.train_label)
        val_y_pred_primitive = np.zeros_like(self.val_label)

        loss_inst = get_xgb_loss_inst(
            list(self.xgb_config.loss_param.keys())[0])
        train_y_pred, val_y_pred = loss_inst.predict(
            train_y_pred_primitive), loss_inst.predict(val_y_pred_primitive)

        for tree_idx in range(1, self.xgb_config.num_trees+1):
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
                self._calc_metrics(self.train_label, train_y_pred, tree_idx, stage="train", loss={
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
            metric = self._calc_metrics(self.val_label, val_y_pred, tree_idx, stage="val",
                                        loss={loss_inst.name: val_loss})
            logger.info("Validation on tree {} done.".format(tree_idx))
            if self.xgb_config.early_stopping_param["patience"] > 0:
                early_stop_flag, save_flag = self.es(metric)
            else:
                early_stop_flag, save_flag = False, True

            if save_flag:
                # self.best_round = tree_idx + 1
                self.best_round = tree_idx
                self.best_prediction_train = copy.deepcopy(train_y_pred)
                self.best_prediction_val = copy.deepcopy(val_y_pred)

            for party_id in FedConfig.get_trainer():
                self.channels["early_stop_com"][party_id].send(early_stop_flag)
            if early_stop_flag:
                logger.info(
                    "label trainer early stopped. best round: {}.".format(self.best_round))
                break

            # if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get("save_frequency") == 0:
            if self.interaction_params.get("save_frequency") > 0 and tree_idx % self.interaction_params.get("save_frequency") == 0:
                # self.save(boosting_tree, epoch=tree_idx+1)
                self.save(boosting_tree, epoch=tree_idx)
                self._write_prediction(
                    # self.train_label, train_y_pred, self.train_ids, epoch=tree_idx + 1)
                    self.train_label, train_y_pred, self.train_ids, epoch=tree_idx)
                self._write_prediction(
                    # self.val_label, val_y_pred, self.val_ids, epoch=tree_idx + 1, stage='val')
                    self.val_label, val_y_pred, self.val_ids, epoch=tree_idx, stage='val')

            # add metrics during training for plot
            self._write_loss(train_loss, val_loss, tree_idx)

        # update the progress of 100 to show the training is finished
        ProgressCalculator.finish_progress()

        # model preserve
        if self.xgb_config.early_stopping_param["patience"] <= 0:
            self.best_round = len(boosting_tree)
            self.best_prediction_train = copy.deepcopy(train_y_pred)
            self.best_prediction_val = copy.deepcopy(val_y_pred)
        else:
            logger.info("num trees: %d, best: %d" % (len(boosting_tree), self.best_round))

        if boosting_tree.trees:
            logger.info("save")
            # self.save(boosting_tree, final=True)
            if self.best_round <= 0:
                self.best_round = len(boosting_tree)
            self.save(boosting_tree[:self.best_round], final=True)
            logger.info('_write_prediction train')
            self._write_prediction(
                self.train_label, train_y_pred, self.train_ids, final=True)
            logger.info('_write_prediction val')
            self._write_prediction(
                self.val_label, val_y_pred, self.val_ids, final=True, stage='val')
            logger.info("Writing roc data...")
            self._write_roc_data(
                self.train_label, train_y_pred, self.val_label, val_y_pred)
            logger.info("Writing ks data...")
            self._write_ks_data(self.train_label, train_y_pred,
                                self.val_label, val_y_pred)
            logger.info("Writing lift and gain data...")
            self._write_lift_gain_data(
                self.train_label, train_y_pred, self.val_label, val_y_pred
            )
            logger.info("Writing pr curve data...")
            self._write_pr_data(
                self.train_label, train_y_pred, self.val_label, val_y_pred)
            self._write_feature_importance()
        else:
            logger.error("Model is none, ture off run_goss (false) and downsampling (1) please.")
            raise SystemError(
                "Model is none, ture off run_goss (false) and downsampling (1) please.")

    def save(self, boosting_tree: BoostingTree, epoch: Optional[int] = None, final: bool = False):
        if final:
            save_model_config(stage_model_config=self.export_conf,
                              save_path=self.output.get("path"))

        save_dir = self.output.get("path")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # dump out ks plot
        suggest_threshold = 0.5
        # if "ks" in self.xgb_config.metrics or "auc_ks" in self.xgb_config.metrics:
        #     # tc = ThresholdCutter(os.path.join(save_dir, "ks_plot_valid.csv"))
        #     tc = ThresholdCutter(os.path.join(
        #         save_dir, self.output.get("ks_plot_val")["name"]))
        #     # tc.cut_by_value(self.val_label, self.best_prediction_val)
        #     # suggest_threshold = float(tc.bst_threshold)
        #     # tc.save()
        #     if final:
        #         self.val_ks_metrics = tc.metrics
        #         self.val_ks_bst_threshold = tc.bst_threshold
        #         self.val_ks_bst_score = tc.bst_score
        #     if self.interaction_params.get("echo_training_metrics"):
        #         tc = ThresholdCutter(os.path.join(
        #             save_dir, self.output.get("ks_plot_train")["name"]))
        #         # tc.cut_by_value(self.train_label, self.best_prediction_train)
        #         # tc.save()
        #         if final:
        #             self.train_ks_metrics = tc.metrics
        #             self.train_ks_bst_threshold = tc.bst_threshold
        #             self.train_ks_bst_score = tc.bst_score
        
        model_name = self.output.get("model", {}).get("name")
        proto_name = self.output.get("proto_model", {}).get("name")
        
        if model_name:
            # model_dict = boosting_tree[:self.best_round].to_dict(
            #     suggest_threshold, compute_group=True)
            model_dict = boosting_tree.to_dict(suggest_threshold, compute_group=True)
            ModelIO.save_json_model(model_dict, save_dir, model_name, epoch=epoch, version='1.4.0')
        
        if proto_name:
            # TODO: temp
            model_name_list = self.output.get("proto_model")["name"].split(".")
            name_prefix, name_postfix = ".".join(
                model_name_list[:-1]), model_name_list[-1]
            if not final and epoch:
                new_model_name = name_prefix + \
                    "_epoch_{}".format(epoch) + "." + name_postfix
            else:
                new_model_name = name_prefix + "." + name_postfix
            model_path = os.path.join(save_dir, new_model_name)

            # xgb_output = boosting_tree[:self.best_round].to_proto(
            #     suggest_threshold, compute_group=True)
            xgb_output = boosting_tree.to_proto(suggest_threshold, compute_group=True)
            
            with open(model_path, 'wb') as f:
                f.write(xgb_output)
            logger.info("model saved as: {}.".format(model_path))
                
        self.make_readable_feature_importance(
            os.path.join(save_dir, self.output.get("feature_importance")["name"]))

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
                    indicator[node_id] = (data <= node.split_info.split_point)
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
        for (owner_name, fid) in tree_feature_importance:
            if owner_name == FedConfig.node_name:
                f_name = self.train_names[fid]
            else:
                f_name = self.remote_f_names[owner_name][fid]
                
            if (owner_name, f_name) not in self.feature_importances_:
                self.feature_importances_[
                    (owner_name, f_name)] = tree_feature_importance[(owner_name, fid)]
            else:
                self.feature_importances_[
                    (owner_name, f_name)] += tree_feature_importance[(owner_name, fid)]
            
            # if (owner_id, fid) not in self.feature_importances_:
            #     self.feature_importances_[
            #         (owner_id, fid)] = tree_feature_importance[(owner_id, fid)]
            # else:
            #     self.feature_importances_[
            #         (owner_id, fid)] += tree_feature_importance[(owner_id, fid)]
        logger.debug("cur feature importance {}".format(
            self.feature_importances_))

    def load_model(self):
        pretrain_path = self.input.get("pretrained_model", {}).get("path", '')
        model_name = self.input.get("pretrained_model", {}).get("name", '')
        
        model_path = Path(
            pretrain_path, model_name
        )
        
        suffix = model_name.split(".")[-1]
        
        if suffix != "pmodel":
            model_dict = ModelIO.load_json_model(model_path)
            boosting_tree = BoostingTree.from_dict(model_dict)
        else:
            with open(model_path, 'rb') as f:
                byte_str = f.read()

            boosting_tree = BoostingTree.from_proto(byte_str)
        return boosting_tree

    def check_dataset(self):
        shapes = self.channels["check_dataset_com"].collect()
        if self.train_dataset is not None:
            m = len(self.train_ids)
            n = len(self.train_features.columns)
            for d in shapes:
                if d["train"][0] != m:
                    raise ValueError(
                        "Lengths of the train set mismatched: %d, %d." % (d["train"][0], m))
                n += d["train"][1]
            if n <= 0:
                raise ValueError(
                    "Number of the feature is zero. Stop training.")

        if self.val_dataset is not None:
            m = len(self.val_ids)
            n = len(self.val_features.columns)
            for d in shapes:
                if d["valid"][0] != m:
                    raise ValueError(
                        "Lengths of the valid set mismatched: %d, %d." % (d["valid"][0], m))
                n += d["valid"][1]
            if n <= 0:
                raise ValueError(
                    "Number of the feature is zero. Stop training.")

        if self.test_dataset is not None:
            m = len(self.test_ids)
            n = len(self.test_features.columns)
            for d in shapes:
                if d["test"][0] != m:
                    raise ValueError(
                        "Lengths of the test set mismatched: %d, %d." % (d["test"][0], m))
                n += d["test"][1]
                if n <= 0:
                    raise ValueError(
                        "Number of the feature is zero. Stop predicting.")
        else:
            if len(shapes) > 0 and "test" in shapes[0]:
                m = shapes[0]["test"][0]
                n = 0
                for d in shapes:
                    if d["test"][0] != m:
                        raise ValueError("Lengths of the test set mismatched.")
                    n += d["test"][1]
                if n <= 0:
                    raise ValueError(
                        "Number of the feature is zero. Stop predicting.")
                else:
                    self.test_dataset = NdarrayIterator(
                        np.zeros((m, 0)), self.bs)
                    self.test_ids = np.arange(m)

    def predict(self):
        out_dict_list = self.channels["sync"].collect()
        self.check_dataset()
        boosting_tree = self.load_model()
        test_y_pred_primitive = self.predict_on_boosting_tree(boosting_tree=boosting_tree,
                                                              data_iterator=self.test_dataset)
        loss_inst = get_xgb_loss_inst(boosting_tree.loss_method)
        test_y_pred = loss_inst.predict(test_y_pred_primitive)
        
        output = {
            "testset": test_y_pred
        }
        
        for out_keys_dict in out_dict_list:
            for key in out_keys_dict:
                if key in output:
                    out_keys_dict[key] = output["testset"]
        self.channels["sync"].scatter(out_dict_list)
        
        save_path = self.output.get("path", '')
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = Path(save_path, self.output.get("testset", {}).get("name", ''))
            if file_path:
                logger.info("predicted results saved at {}".format(file_path))
                pd.DataFrame({"id": self.test_ids, "pred": test_y_pred}).to_csv(
                    file_path, float_format="%.6g", index=False, header=True
                )
