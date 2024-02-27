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
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from algorithm.core.data_io import CsvReader, NdarrayIterator
from algorithm.core.tree.tree_structure import BoostingTree, Tree
from algorithm.core.tree.cat_param_parser import parse_category_param
from algorithm.core.tree.xgboost_loss import get_xgb_loss_inst
# from algorithm.framework.vertical.kmeans.api import get_table_agg_trainer_inst
from algorithm.core.horizontal.aggregation.api import get_aggregation_leaf_inst
from algorithm.core.tree.feature_importance import FeatureImportance
from common.utils.utils import save_model_config
from common.utils.logger import logger
from common.utils.algo_utils import earlyStoppingH
from common.communication.gRPC.python.channel import DualChannel
from common.evaluation.metrics import BiClsMetric, DecisionTable, ThresholdCutter
from service.fed_config import FedConfig
from .decision_tree_label_trainer import HorizontalDecisionTreeLabelTrainer
from .common import Common
from common.evaluation.metrics import LiftGainCalculator


class HorizontalXgboostLabelTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        self.__init_data()
        self.__init_channels()
        self.__init_cat_columns()
        self.feature_importances_ = {}
        self.es = earlyStoppingH(
            key=self.common_config.early_stopping.get("key", "ks"), 
            patience=self.common_config.early_stopping.get("patience", -1), 
            delta=self.common_config.early_stopping.get("delta", 0)
        )
        self.best_round = -1
        self.best_prediction_val = None
        self.best_prediction_train = None
        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.common_config.identity,
            "filename": self.common_config.output.get("model")["name"]
        }]

    def __init_channels(self):
        self.agg_inst = get_aggregation_leaf_inst(
            sec_conf=self.encryption_params,
            root_id=FedConfig.get_assist_trainer(),
            leaf_ids=FedConfig.get_label_trainer() + FedConfig.get_trainer()
        )
        self.col_sample_channel = DualChannel(
            name="col_sample_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.cat_columns_channel = DualChannel(
            name="cat_columns_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.bin_split_points_channel = DualChannel(
            name="bin_split_points_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.best_split_info_channel = DualChannel(
            name="best_split_info_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.sample_split_channel = DualChannel(
            name="sample_split_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.tree_structure_channel = DualChannel(
            name="tree_structure_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.restart_channel = DualChannel(
            name="restart_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.early_stop_channel = DualChannel(
            name="early_stop_" + FedConfig.node_id, 
            ids=[FedConfig.node_id, FedConfig.get_assist_trainer()]
        )
        self.channels = {
            "agg_inst": self.agg_inst,
            "best_split_info_channel": self.best_split_info_channel,
            "sample_split_channel": self.sample_split_channel,
            "tree_structure_channel": self.tree_structure_channel,
            "restart_channel": self.restart_channel,
            "early_stop_channel": self.early_stop_channel,
        }

    def __init_data(self) -> None:
        """ Init data, include features and label.

        Returns: None

        """
        self.bs = self.common_config.train_params.get("val_batch_size")
        if self.common_config.input_trainset:
            self.train_features, self.train_label, self.train_ids, self.train_names = \
                self.__load_data(self.common_config.input_trainset)
            self.train_dataset = NdarrayIterator(self.train_features.to_numpy(), self.bs)
        else:
            self.train_dataset = None
        if self.common_config.input_valset:
            self.val_features, self.val_label, self.val_ids, self.val_names = \
                self.__load_data(self.common_config.input_valset)
            self.val_dataset = NdarrayIterator(self.val_features.to_numpy(), self.bs)
        else:
            self.val_dataset = None

        if self.train_dataset:
            self.feature_names = self.train_features.columns.to_list()
        else:
            self.train_features = None

    def __init_cat_columns(self):
        cat_columns = parse_category_param(
            self.train_features,
            col_index=self.xgb_config.cat_col_index,
            col_names=self.xgb_config.cat_col_names,
            col_index_type=self.xgb_config.cat_col_index_type,
            col_names_type=self.xgb_config.cat_col_names_type
        )
        self.cat_columns = cat_columns
        self.cat_feature_names = []

        # if len(cat_columns) > 0:
        #     self.cat_feature_names = self.train_features.columns[cat_columns].to_list()

        self.cat_columns_channel.send(self.cat_columns)

    def __load_data(self, config):
        """ Load data from dataset config.

        Args:
            config: Dataset config.

        Returns: [CsvReader, ...]

        """
        if len(config) > 1:
            logger.warning("More than one dataset is not supported.")

        if not config:
            return None, None, None

        config = config[0]
        if config["type"] == "csv":
            path = Path(config["path"], config["name"])
            if not path:
                return None, None, None
            data_reader = CsvReader(path, has_id=config["has_id"], has_label=config["has_label"])
            features = data_reader.features(type="pandas.dataframe")
            features = features.fillna(0).replace({self.xgb_config.missing_value: 0})
            ids = data_reader.ids
            names = data_reader.feature_names()
            labels = data_reader.label()
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return features, labels, ids, names

    def _feature_binning(self):
        """
        apply horizontal feature binning

        1. label trainer calculates local split points
        2. assist trainer average split points
        3. label trainer convert features with the results

        denotes:
            train_features - R ^ (m * n)
            num_bins - k

        then:
            self.split_points - list of length n, with each element has length of (k - 1)
            *the first and last element of split points are -np.inf and np.inf

        """
        if self.xgb_config.num_bins < 2:
            raise ValueError("parameter num_bins must be larger than 2.")

        self._continuous_feature_binning()
        # if len(self.cat_columns) > 0:
        #     self._cat_feature_binning()

        self._convert_to_binned_data()
    
    # def _cat_feature_binning(self):
    #     local_cat_max_values = self.train_features[self.cat_feature_names].max().to_numpy()
    #     self.cat_columns_channel.send(local_cat_max_values)
    #     global_cat_max_values = self.cat_columns_channel.recv()
    #     self.cat_bins = dict()
    #     for i, feature_name in enumerate(self.cat_feature_names):
    #         value_counts = np.zeros(int(global_cat_max_values[i]) + 1)
    #         vc = self.train_features[feature_name].value_counts()
    #         value_counts[vc.index.to_list()] = vc.values
    #         self.agg_inst.send(torch.Tensor(value_counts))
    #         bins = self.cat_columns_channel.recv()
    #         self.cat_bins[feature_name] = bins
            
    def _continuous_feature_binning(self):
        def f(x: pd.Series):
            return [x.min(), x.max()]

        features = [_ for _ in self.train_features.columns if _ not in self.cat_feature_names]
        split_mat = self.train_features[features].apply(f).to_numpy()
        self.bin_split_points_channel.send(split_mat)
        self.split_points = self.bin_split_points_channel.recv()

    # def _continuous_feature_binning(self):
    #     def f(x):
    #         _, split_points = pd.cut(x, bins=self.xgb_config.num_bins, retbins=True,
    #                                  right=False, labels=range(self.xgb_config.num_bins))
    #         return split_points[1:-1]

    #     features = [_ for _ in self.train_features.columns if _ not in self.cat_feature_names]

    #     split_mat = self.train_features[features].apply(f).to_numpy()
    #     self.agg_inst.send(torch.Tensor(split_mat))
    #     split_mat = self.bin_split_points_channel.recv()
    #     self.split_points = split_mat.T.tolist()

    def _convert_to_binned_data(self):
        """
        convert train features to binned data

        Returns:

        """

        # if len(self.cat_columns) > 0:
        #     self.train_features[self.cat_feature_names] = self.train_features[self.cat_feature_names].astype('category')

        def trans(x):
            # if x[1] in self.cat_feature_names:
            #     value_map = {i: k for i, k in enumerate(self.cat_bins[x[1]])}
            #     codes = self.train_features[x[1]].map(value_map).fillna(self.xgb_config.num_bins - 1)
            #     return pd.Series(codes, name=x[1])
            # else:
            bins = self.split_points[self.con_idx_mapping.get(x[0])]
            bins = np.insert(bins, 0, -np.inf)
            bins = np.insert(bins, len(bins), np.inf)
            return pd.cut(
                self.train_features[x[1]], bins=bins, labels=range(self.xgb_config.num_bins)
            )

        con_idx = [i for i, f in enumerate(self.train_features.columns) if f not in self.cat_feature_names]
        self.con_idx_mapping = {k: v for v, k in enumerate(con_idx)}
        out = pd.Series(enumerate(self.train_features.columns)).apply(trans).T
        
        if self.xgb_config.num_bins <= 256:
            dtype = np.uint8
        elif self.xgb_config.num_bins <= 2 ** 16:
            dtype = np.uint16
        else:
            dtype = np.uint32

        self.train_features = out.astype(dtype)

    def _col_sample(self):
        n = len(self.train_features.columns)
        self.col_sample_channel.send(n)
        feature_id_mapping = self.col_sample_channel.recv()
        sampled_idx = sorted(list(feature_id_mapping.values()))
        sampled_features = self.train_features.iloc[:, sampled_idx]
        return sampled_features, feature_id_mapping

    def update_feature_importance(self, tree_feature_importance):
        for (owner_name, fid) in tree_feature_importance:
            f_name = self.train_names[fid]
                
            if (owner_name, f_name) not in self.feature_importances_:
                self.feature_importances_[
                    (owner_name, f_name)] = tree_feature_importance[(owner_name, fid)]
            else:
                self.feature_importances_[
                    (owner_name, f_name)] += tree_feature_importance[(owner_name, fid)]
                
        logger.debug("cur feature importance {}".format(
            self.feature_importances_))

    def fit(self):
        logger.info("HorizontalXGBoost::LabelTrainer start.")
        self._feature_binning()

        self.boosting_tree = BoostingTree()

        self.train_y_pred_primitive = np.zeros_like(self.train_label)
        self.val_y_pred_primitive = np.zeros_like(self.val_label)

        self.loss_inst = get_xgb_loss_inst(list(self.xgb_config.loss_param.keys())[0])
        self.train_y_pred = self.loss_inst.predict(self.train_y_pred_primitive)

        for tree_idx in range(self.xgb_config.num_trees):
            while True:
                # train a tree until restart != 1
                trainer, restart_status = self.train_loop(tree_idx)
                if restart_status != 1:
                    break
                logger.info(f"label trainer tree {tree_idx} training restart.")

            if restart_status == 2:
                logger.info("label trainer early stopped because a tree's root is leaf, best round: {}.".format(
                    self.best_round))
                break

            self.update_feature_importance(trainer.feature_importance)
            self.cal_train_metric(tree_idx)
            early_stop_flag = self.val_loop(tree_idx)
            if early_stop_flag:
                break

    def train_loop(self, tree_idx):
        restart_status = 1
        sampled_features, feature_id_mapping = self._col_sample()
        cat_columns_after_sampling = list(filter(
            lambda x: feature_id_mapping[x] in self.cat_columns, list(feature_id_mapping.keys())))
        cat_bins_after_sampling = [self.cat_bins[self.feature_names[v]] for v in
                                   feature_id_mapping.values() if v in self.cat_columns]

        split_points_after_sampling = [
            self.split_points[
                self.con_idx_mapping[feature_id_mapping[k]]] for k in feature_id_mapping.keys()
            if feature_id_mapping[k] in self.con_idx_mapping
        ]
        logger.info("HorizontalXGBoost::LabelTrainer::Tree {} start training.".format(tree_idx))
        trainer = HorizontalDecisionTreeLabelTrainer(
            tree_param=self.xgb_config,
            y=self.train_label,
            y_pred=self.train_y_pred,
            features=sampled_features,
            cat_columns=cat_columns_after_sampling,
            cat_bins=cat_bins_after_sampling,
            split_points=split_points_after_sampling,
            channels=self.channels,
            tree_index=tree_idx
        )

        self.tree = trainer.fit()
        if not self.tree.root_node.is_leaf:
            restart_status = 0
        else:
            if self.xgb_config.early_stopping_param["patience"] <= 0:
                # if not set patience, terminate immediately
                restart_status = 2
            else:
                self.es.counter += 1

        if self.es.counter == self.xgb_config.early_stopping_param["patience"]:
            restart_status = 2

        self.channels["restart_channel"].send(restart_status)
        return trainer, restart_status

    def cal_train_metric(self, tree_idx):
        # run_goss is not supported now
        for _, node in self.tree.nodes.items():
            if node.is_leaf:
                self.train_y_pred_primitive[node.sample_index] += node.weight * \
                    self.xgb_config.learning_rate

        self.train_y_pred = self.loss_inst.predict(self.train_y_pred_primitive)
        if self.echo_training_metrics:
            train_loss = self.loss_inst.cal_loss(
                self.train_label, self.train_y_pred_primitive, after_prediction=False)
            self._calc_metrics(self.train_label, self.train_y_pred, tree_idx, stage="train", loss={
                self.loss_inst.name: train_loss})

        self.tree.clear_training_info()
        self.boosting_tree.append(
            tree=self.tree,
            lr=self.xgb_config.learning_rate,
            max_depth=self.xgb_config.max_depth
        )
        logger.info("HorizontalXGBoost::LabelTrainer::Tree {} training done.".format(tree_idx))

    def val_loop(self, tree_idx):
        logger.info("HorizontalXGBoost::LabelTrainer::Validation on tree {} start.".format(tree_idx))
        self.val_y_pred_primitive += self.predict_on_tree(self.tree, self.val_dataset) * self.xgb_config.learning_rate
        val_y_pred = self.loss_inst.predict(self.val_y_pred_primitive)
        val_loss = self.loss_inst.cal_loss(self.val_label, self.val_y_pred_primitive, after_prediction=False)
        logger.info("HorizontalXGBoost::LabelTrainer::tree %d val_loss: %.3f." % (tree_idx, val_loss))
        metric = self._calc_metrics(self.val_label, val_y_pred, tree_idx, stage="val", loss={self.loss_inst.name: val_loss})
        logger.info("HorizontalXGBoost::LabelTrainer::Validation on tree {} done.".format(tree_idx))

        if self.xgb_config.early_stopping_param["patience"] > 0:
            early_stop_flag, save_flag = self.es(metric)
        else:
            early_stop_flag, save_flag = False, True

        if save_flag:
            self.best_round = tree_idx + 1
            self.best_prediction_train = copy.deepcopy(self.train_y_pred)
            self.best_prediction_val = copy.deepcopy(val_y_pred)

        self.channels["early_stop_channel"].send(early_stop_flag)
        
        if early_stop_flag:
            logger.info("label trainer early stopped. best round: {}.".format(self.best_round))
            return early_stop_flag

        if self.save_frequency > 0 and (tree_idx + 1) % self.save_frequency == 0:
            self.save(self.boosting_tree, epoch=tree_idx+1)
            self._write_prediction(self.train_label, self.train_y_pred, self.train_ids, epoch=tree_idx + 1)
            self._write_prediction(self.val_label, val_y_pred, self.val_ids, epoch=tree_idx + 1, stage='val')
            
            logger.info("Writing roc data...")
            self._write_roc_data(
                self.train_label, self.train_y_pred, self.val_label, val_y_pred)
            logger.info("Writing ks data...")
            self._write_ks_data(self.train_label, self.train_y_pred,
                                self.val_label, val_y_pred)
            logger.info("Writing lift and gain data...")
            self._write_lift_gain_data(
                self.train_label, self.train_y_pred, self.val_label, val_y_pred
            )
            logger.info("Writing pr curve data...")
            self._write_pr_data(
                self.train_label, self.train_y_pred, self.val_label, val_y_pred)
            self._write_feature_importance()
        return early_stop_flag

    def save(self, boosting_tree: BoostingTree, epoch: Optional[int] = None, final: bool = False):
        if final:
            save_model_config(stage_model_config=self.export_conf, save_path=self.save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # dump out ks plot
        suggest_threshold = 0.5
        if "ks" in self.xgb_config.metrics or "auc_ks" in self.xgb_config.metrics:
            # tc = ThresholdCutter(os.path.join(save_dir, "ks_plot_valid.csv"))
            tc = ThresholdCutter(os.path.join(self.save_dir, self.common_config.output.get("ks_plot_val")["name"]))
            tc.cut_by_value(self.val_label, self.best_prediction_val)
            suggest_threshold = float(tc.bst_threshold)
            tc.save()
            if self.echo_training_metrics:
                tc = ThresholdCutter(os.path.join(self.save_dir, self.common_config.output.get("ks_plot_train")["name"]))
                tc.cut_by_value(self.train_label, self.best_prediction_train)
                tc.save()

        model_name_list = self.common_config.output.get("model")["name"].split(".")
        name_prefix, name_postfix = ".".join(model_name_list[:-1]), model_name_list[-1]
        if not final and epoch:
            model_name = name_prefix + "_epoch_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix
        model_path = os.path.join(self.save_dir, model_name)

        xgb_output = boosting_tree[:self.best_round].to_proto(
            suggest_threshold, compute_group=True)
        with open(model_path, 'wb') as f:
            f.write(xgb_output)
        logger.info("model saved as: {}.".format(model_path))

        self.make_readable_feature_importance(os.path.join(
                self.save_dir, 
                self.common_config.output.get("feature_importance")["name"]
            ))

    def make_readable_feature_importance(self, file_name):
        with open(file_name, "w") as f:
            f.write("owner_id,fid,importance\n")
            normalizer = np.sum([_.get() for _ in self.feature_importances_.values()])
            for k, v in sorted(self.feature_importances_.items(), key=lambda d: d[1], reverse=True):
                f.write("%s,%s,%.6g\n" % (k[0], k[1], v.get() / normalizer))

    def predict_on_tree(self, tree: Tree, data_iterator: NdarrayIterator) -> np.ndarray:
        prediction = np.zeros(len(data_iterator.data), dtype=np.float32)
        for i, data in enumerate(data_iterator):
            indicator = self._make_indicator_for_prediction(tree, data)
            prediction[i * data_iterator.bs: (i + 1) * data_iterator.bs] = \
                  self._gen_prediction(tree, indicator, data)
        return prediction

    @staticmethod
    def _make_indicator_for_prediction(tree: Tree, feature: np.ndarray):
        indicator = {}
        for node_id, node in tree.nodes.items():
            if not node.is_leaf:
                feature_idx = node.split_info.feature_idx
                data = feature[:, feature_idx]
                if node.split_info.is_category:
                    indicator[node_id] = np.isin(data, node.split_info.left_cat)
                else:
                    indicator[node_id] = (data < node.split_info.split_point)
        return indicator

    @staticmethod
    def _gen_prediction(tree: Tree, indicator: Dict[str, np.ndarray], feature: np.ndarray):
        prediction = np.zeros(len(feature), dtype=np.float32)
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

    def _calc_metrics(self, y, p, epoch, stage="train", loss={}):
        if stage == "train" and not self.echo_training_metrics:
            return
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        output_file = os.path.join(
            self.metric_dir, 
            self.common_config.output.get("metric_" + stage)["name"]
        )
        if loss:
            evaluate = BiClsMetric(epoch, output_file, self.xgb_config.metrics)
        else:
            evaluate = BiClsMetric(epoch, output_file, self.xgb_config.metrics, self.xgb_config.loss_param)

        evaluate.calc_metrics(y, p)
        for key, value in loss.items():
            evaluate.metrics[key] = value
        evaluate.save()
        if "decision_table" in self.xgb_config.metrics:
            dt = DecisionTable(self.xgb_config.metrics["decision_table"])
            dt.fit(y, p)
            dt.save(os.path.join(
                self.metric_dir, 
                self.common_config.output.get("decision_table_" + stage)["name"]
            ))
        logger.info("{} {}".format(stage, evaluate))
        return evaluate.metrics

    def _write_prediction(self, y, p, idx=None, epoch=None, final=False, stage="train"):
        if stage == "train" and not self.write_training_prediction:
            return
        elif stage == "val" and not self.write_validation_prediction:
            return
        elif stage not in ("train", "val"):
            raise ValueError("stage must be 'train' or 'val'.")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if final:
            file_name = os.path.join(
                self.save_dir, self.common_config.output.get("prediction_" + stage)["name"])
        else:
            file_name = self.common_config.output.get("prediction_" + stage)["name"]
            file_name_list = file_name.split(".")
            # file_name_list.insert(-1, '_epoch_'+str(epoch))
            file_name_list[-2] += '_epoch_' + str(epoch)
            file_name = '.'.join(file_name_list)
            file_name = os.path.join(self.save_dir, file_name)
        if idx is None:
            df = pd.DataFrame({"pred": p, "label": y})
            df = df.reset_index().rename(columns={"index": "id"})
            df.to_csv(file_name, header=True, index=False, float_format='%.6g')
        else:
            df = pd.DataFrame({'id': idx, "pred": p, "label": y})
            df.to_csv(file_name, header=True, index=False, float_format='%.6g')
            
    def _write_roc_data(self, train_label, train_pred, val_label=None, val_pred=None, n_cuts=1000) -> None:

        # prepare write path
        try:
            file_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_roc']['name']
            )
        except Exception:
            file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_roc.json"
            )

        try:
            train_fpr, train_tpr, _ = roc_curve(train_label, train_pred)
        except Exception:
            logger.error("Could not calculate train roc curve")
            return

        # pruning
        size = train_fpr.size
        if size > n_cuts:
            logger.info(
                f"Too much points in training roc curve. Cutting to {n_cuts} points")
            train_fpr = self.__prune_data_size(train_fpr, n_cuts)
            train_tpr = self.__prune_data_size(train_tpr, n_cuts)

        # add to roc_list
        roc_list = []
        for fpr, tpr in zip(train_fpr, train_tpr):
            roc_list.append(
                {"fpr": round(fpr, 6), "tpr": round(tpr, 6), "period": "train"})

        # If val data exists
        if val_label is not None:
            try:
                val_fpr, val_tpr, _ = roc_curve(val_label, val_pred)
            except Exception:
                logger.error("Could not calculate val roc curve")
                with open(file_path, "w") as file:
                    json.dump(roc_list, file)
                return

            # pruning
            if val_fpr.size > n_cuts:
                logger.info(
                    f"Too much points in validation roc curve. Cutting to {n_cuts} points")
                val_fpr = self.__prune_data_size(val_fpr, n_cuts)
                val_tpr = self.__prune_data_size(val_tpr, n_cuts)

            for fpr, tpr in zip(val_fpr, val_tpr):
                roc_list.append(
                    {"fpr": round(fpr, 6), "tpr": round(tpr, 6), "period": "val"})

        # Sort
        logger.info("Sorting roc list")
        roc_list = sorted(roc_list, key=lambda el: el['fpr'])

        logger.info("Writing roc to file")
        with open(file_path, "w") as file:
            json.dump(roc_list, file)
        return
    
    def _write_ks_data(self, train_label, train_pred, val_label=None, val_pred=None, n_cuts=1000) -> None:
        # Setup file path
        try:
            file_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_ks']['name']
            )
        except Exception:
            file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_ks.json"
            )

        ks_list = []
        tc = ThresholdCutter()

        # Train
        # if self.train_ks_metrics is None:
        if True:
            tc.sim_cut_by_value(train_label, train_pred)
            train_ks_df = pd.DataFrame(tc.metrics)
            # train_bst_threshold = tc.bst_threshold
            # train_bst_score = tc.bst_score
        else:
            logger.info("Using calculated train ks")
            train_ks_df = pd.DataFrame(self.train_ks_metrics)
            # train_bst_threshold = self.train_ks_bst_threshold
            # train_bst_score = self.train_ks_bst_score

        # pruning
        if train_ks_df.shape[0] > n_cuts:
            logger.info(
                f"Pruning training ks data. Before pruning: {train_ks_df.shape[0]}")
            prune_train_ks_df = pd.DataFrame()
            prune_train_ks_df['threshold'] = self.__prune_data_size(
                train_ks_df['threshold'].values, n_cuts)
            prune_train_ks_df['tpr'] = self.__prune_data_size(
                train_ks_df['tpr'].values, n_cuts)
            prune_train_ks_df['fpr'] = self.__prune_data_size(
                train_ks_df['fpr'].values, n_cuts)
            train_ks_df = prune_train_ks_df
            logger.info(
                f"After pruning, training ks data: {train_ks_df.shape[0]}")

        for _, row in train_ks_df.iterrows():
            ks_list.append(
                {"thres": round(row['threshold'], 6), "value": round(row['tpr'], 6), "type_period": "tpr_train"})
            ks_list.append(
                {"thres": round(row['threshold'], 6), "value": round(row['fpr'], 6), "type_period": "fpr_train"})
            ks_list.append(
                {"thres": round(row['threshold'], 6), "value": round(row['ks'], 6), "type_period": "ks_train"})

        # Val
        if val_label is not None:
            # if self.val_ks_metrics is None:
            if True:
                tc.sim_cut_by_value(val_label, val_pred)
                val_ks_df = pd.DataFrame(tc.metrics)
                # val_bst_threshold = tc.bst_threshold
                # val_bst_score = tc.bst_score
            else:
                logger.info(f"Using calculated val ks")
                val_ks_df = pd.DataFrame(self.val_ks_metrics)
                # val_bst_threshold = self.val_ks_bst_threshold
                # val_bst_score = self.val_ks_bst_score

            # pruning
            if val_ks_df.shape[0] > n_cuts:
                logger.info(
                    f"Pruning val ks data. Before pruning: {val_ks_df.shape[0]}")
                prune_val_ks_df = pd.DataFrame()
                prune_val_ks_df['threshold'] = self.__prune_data_size(
                    val_ks_df['threshold'].values, n_cuts)
                prune_val_ks_df['tpr'] = self.__prune_data_size(
                    val_ks_df['tpr'].values, n_cuts)
                prune_val_ks_df['fpr'] = self.__prune_data_size(
                    val_ks_df['fpr'].values, n_cuts)
                val_ks_df = prune_val_ks_df
                logger.info(
                    f"After pruning, val ks data: {val_ks_df.shape[0]}")

            for _, row in val_ks_df.iterrows():
                ks_list.append(
                    {"thres": round(row['threshold'], 6), "value": round(row['tpr'], 6), "type_period": "tpr_val"})
                ks_list.append(
                    {"thres": round(row['threshold'], 6), "value": round(row['fpr'], 6), "type_period": "fpr_val"})
                ks_list.append(
                    {"thres": round(row['threshold'], 6), "value": round(row['ks'], 6), "type_period": "ks_val"})

        # Sort lists
        logger.info("Sorting ks list")
        ks_list = sorted(ks_list, key=lambda el: el["thres"])

        logger.info("Writing ks to file")
        with open(file_path, "w") as file:
            json.dump(ks_list, file)

        return
    
    def _write_lift_gain_data(self, train_label, train_pred, val_label=None, val_pred=None) -> None:
        # Setup file path
        try:
            lift_file_path = Path(
                self.metric_dir, self.train_conf['output']['plot_lift']['name']
            )
        except Exception:
            lift_file_path = Path(
                self.metric_dir, f"{self.model_info['name']}_plot_lift.json"
            )
        try:
            gain_file_path = Path(
                self.metric_dir, self.train_conf['output']['plot_gain']['name']
            )
        except Exception:
            gain_file_path = Path(
                self.metric_dir, f"{self.model_info['name']}_plot_gain.json"
            )

        lift_gain_cal = LiftGainCalculator()
        lift_gain_cal.cal_lift_gain(train_label, train_pred)
        train_lift_gain_df = lift_gain_cal.metrics
        train_lift_gain_df = train_lift_gain_df.query("lift.notnull()")
        # cut_list = []
        gain_list = []
        lift_list = []
        for _, row in train_lift_gain_df.iterrows():
            gain_list.append(
                {
                    "bin_val": round(row['percentage_data'], 6),
                    "gain": round(row['cum_gain'], 6),
                    "period": "train"
                }
            )

            lift_list.append(
                {
                    "bin_val": round(row['percentage_data'], 6),
                    "lift": round(row['lift'], 6),
                    "period": "train"
                }
            )
        logger.info(f"Training lift point number: {len(lift_list)}")

        if val_label is not None:
            lift_gain_cal.cal_lift_gain(val_label, val_pred)
            val_lift_gain_df = lift_gain_cal.metrics
            val_lift_gain_df = val_lift_gain_df.query("lift.notnull()")

            for _, row in val_lift_gain_df.iterrows():
                gain_list.append(
                    {
                        "bin_val": round(row['percentage_data'], 6),
                        "gain": round(row['cum_gain'], 6),
                        "period": "val"
                    }
                )

                lift_list.append(
                    {
                        "bin_val": round(row['percentage_data'], 6),
                        "lift": round(row['lift'], 6),
                        "period": "val"
                    }
                )
            logger.info(f"Val lift point number: {len(lift_list)}")

        # Sort values by horizontal axis
        logger.info("Sorting gain and list lists")
        gain_list = sorted(gain_list, key=lambda el: el["bin_val"])
        lift_list = sorted(lift_list, key=lambda el: el["bin_val"])

        with open(lift_file_path, "w") as lift_file:
            json.dump(lift_list, lift_file)

        with open(gain_file_path, "w") as gain_file:
            json.dump(gain_list, gain_file)

        return
    
    @staticmethod
    def __prune_data_size(data: np.array, n_cuts: int):
        size = data.size
        cuts = np.linspace(0, 1, n_cuts + 1)
        index_list = [int(size * cut) for cut in cuts]
        if index_list[-1] >= size:
            index_list = index_list[:-1]
        return data[index_list]
    
    def _write_pr_data(self, train_label, train_pred, val_label=None, val_pred=None, n_cuts=1000) -> None:
        # Set up file path
        try:
            pr_curve_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_precision_recall']['name']
            )
        except Exception:
            pr_curve_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_pr_curve.json"
            )

        pr_list = []
        # Add train pr data
        precision, recall, thres = precision_recall_curve(
            train_label, train_pred)

        # pruning
        if precision.size > n_cuts:
            logger.info("Too much points in training pr curve, pruning")
            precision = self.__prune_data_size(precision, n_cuts)
            recall = self.__prune_data_size(recall, n_cuts)

        for pr, rc in zip(precision, recall):
            pr_list.append(
                {
                    "recall": round(rc, 6),
                    "precision": round(pr, 6),
                    "period": "train"
                }
            )

        # Add val pr data
        if val_label is not None:
            precision, recall, thres = precision_recall_curve(
                val_label, val_pred)

            # pruning
            if precision.size > n_cuts:
                logger.info("Too much points in validation pr curve, pruning")
                precision = self.__prune_data_size(precision, n_cuts)
                recall = self.__prune_data_size(recall, n_cuts)

            for pr, rc in zip(precision, recall):
                pr_list.append(
                    {
                        "recall": round(rc, 6),
                        "precision": round(pr, 6),
                        "period": "val"
                    }
                )

        # Sort
        logger.info("Sorting pr list")
        pr_list = sorted(pr_list, key=lambda el: el["recall"])

        # Write file
        with open(pr_curve_path, "w") as pr_file:
            json.dump(pr_list, pr_file)

        return
    
    def _write_feature_importance(self):
        # Get importance file path
        try:
            feature_importance_file_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_feature_importance']['name']
            )
        except Exception:
            feature_importance_file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_feature_importance.json"
            )

        logger.info(f"Feature importances: {self.feature_importances_}")

        feature_importance_list = []
        # Get and normalize feature importance
        try:
            normalizer = np.sum([_.get()
                                 for _ in self.feature_importances_.values()])
        except Exception:
            normalizer = np.sum(
                [abs(_) for _ in self.feature_importances_.values()])

        for k, v in sorted(self.feature_importances_.items(), key=lambda d: d[1], reverse=True):
            feature_name = "_".join(map(str, k))
            if isinstance(v, FeatureImportance):
                feature_importance = v.get() / normalizer
            else:
                feature_importance = v / normalizer
            feature_importance_list.append(
                {
                    "feature": feature_name,
                    "importance": round(feature_importance, 6)
                }
            )

        feature_importance_list = sorted(
            feature_importance_list, key=lambda d: abs(d['importance']), reverse=True
        )

        # Write file
        with open(feature_importance_file_path, "w") as feature_importance_file:
            json.dump(feature_importance_list, feature_importance_file)
