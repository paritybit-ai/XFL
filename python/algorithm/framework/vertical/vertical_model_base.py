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
from pathlib import Path
import pandas as pd
import numpy as np

from common.evaluation.metrics import BiClsMetric, DecisionTable, RegressionMetric, ThresholdCutter, LiftGainCalculator
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from sklearn.metrics import roc_curve
import json
from sklearn.metrics import precision_recall_curve
from algorithm.core.tree.feature_importance import FeatureImportance
from algorithm.core.tree.feature_importance import FeatureImportance


class VerticalModelBase(TrainConfigParser):
    def __init__(self, train_conf: dict, label: bool = False):
        super().__init__(train_conf)
        self._parse_config()
        self.train_conf = train_conf
        self.label = label

        self.train_ks_metrics = None
        self.val_ks_metrics = None

    def _parse_config(self) -> None:
        # output_path
        # self.save_dir = Path(self.output.get("model", {}).get("path", ""))
        self.save_dir = Path(self.output.get("path", ""))
        self.metric_dir = self.save_dir
        # params
        self.lossfunc_conifg = self.train_params.get("lossfunc", {})
        self.metric_config = self.train_params.get("metric", {})
        # interaction_params
        self.echo_training_metrics = self.interaction_params.get(
            "echo_training_metrics", False)
        self.write_training_prediction = self.interaction_params.get(
            "write_training_prediction", False)
        self.write_validation_prediction = self.interaction_params.get(
            "write_validation_prediction", False)
        # if self.output.get("metrics"):
        # 	self.metric_path = Path(self.output["metrics"].get("path"))
        # else:
        # 	self.metric_path = self.save_dir
        # # params
        # self.lossfunc_conifg = self.train_params.get("lossfunc_config")
        # self.metric_config = self.train_params.get("metric_config")
        # # interaction_params
        # self.echo_training_metrics = self.interaction_params.get("echo_training_metrics")
        # self.write_training_prediction = self.interaction_params.get("write_training_prediction")
        # self.write_validation_prediction = self.interaction_params.get("write_validation_prediction")

    def _calc_metrics(self, y, p, epoch, stage="train", loss={}):
        if stage == "train" and not self.echo_training_metrics:
            return
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)
        # output_file = os.path.join(
        #     self.metric_path, "{}_metrics.csv".format(stage))

        output_file = os.path.join(
            self.metric_dir, self.output.get("metric_" + stage)["name"])
        if self.model_info["name"] not in ["vertical_linear_regression", "vertical_poisson_regression"]:
            if loss:
                evaluate = BiClsMetric(epoch, output_file, self.metric_config)
            else:
                evaluate = BiClsMetric(
                    epoch, output_file, self.metric_config, self.lossfunc_conifg)
        else:
            evaluate = RegressionMetric(epoch, output_file, self.metric_config)
        evaluate.calc_metrics(y, p)
        for key, value in loss.items():
            evaluate.metrics[key] = value
        if self.model_info["name"] not in ["vertical_linear_regression", "vertical_poisson_regression"]:
            evaluate.save()
        else:
            evaluate.save(evaluate.metrics)
        if "decision_table" in self.metric_config:
            dt = DecisionTable(self.metric_config["decision_table"])
            dt.fit(y, p)
            dt.save(os.path.join(self.metric_dir, self.output.get(
                "decision_table_" + stage)["name"]))
            # dt.save(os.path.join(self.metric_path,
            #         "{}_decision_table.csv".format(stage)))
        logger.info("{} {}".format(stage, evaluate))
        return evaluate.metrics

    def _write_loss(self, train_loss, val_loss, epoch):
        # prepare write path
        try:
            file_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_loss']['name']
            )
        except:
            file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_loss.json"
            )

        if file_path.is_file():
            with open(file_path, "r") as file:
                prev_loss_list = json.load(file)
        else:
            prev_loss_list = []

        prev_loss_list.append(
            {'epoch': epoch, 'loss': train_loss, 'period': 'train'})
        prev_loss_list.append(
            {'epoch': epoch, 'loss': val_loss, 'period': 'val'})

        with open(file_path, "w") as out_fp:
            json.dump(prev_loss_list, out_fp)

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
            # file_name = os.path.join(self.save_dir, "predicted_probabilities_{}.csv".format(stage))
            file_name = os.path.join(
                self.save_dir, self.output.get("prediction_" + stage)["name"])
        else:
            # file_name = os.path.join(self.save_dir, "predicted_probabilities_{}.epoch_{}".format(stage, epoch))
            file_name = self.output.get("prediction_" + stage)["name"]
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

    def _write_plot_data(self):
        pass

    @staticmethod
    def __prune_data_size(data: np.array, n_cuts: int):
        size = data.size
        cuts = np.linspace(0, 1, n_cuts + 1)
        index_list = [int(size * cut) for cut in cuts]
        if index_list[-1] >= size:
            index_list = index_list[:-1]
        return data[index_list]

    def _write_roc_data(self, train_label, train_pred, val_label=None, val_pred=None, n_cuts=1000) -> None:

        # prepare write path
        try:
            file_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_roc']['name']
            )
        except:
            file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_roc.json"
            )

        try:
            train_fpr, train_tpr, _ = roc_curve(train_label, train_pred)
        except:
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
        if not val_label is None:
            try:
                val_fpr, val_tpr, _ = roc_curve(val_label, val_pred)
            except:
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
        except:
            file_path = Path(
                self.metric_dir,
                f"{self.model_info['name']}_plot_ks.json"
            )

        ks_list = []
        tc = ThresholdCutter()

        # Train
        if self.train_ks_metrics is None:
            tc.sim_cut_by_value(train_label, train_pred)
            train_ks_df = pd.DataFrame(tc.metrics)
            train_bst_threshold = tc.bst_threshold
            train_bst_score = tc.bst_score
        else:
            logger.info(f"Using calculated train ks")
            train_ks_df = pd.DataFrame(self.train_ks_metrics)
            train_bst_threshold = self.train_ks_bst_threshold
            train_bst_score = self.train_ks_bst_score

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
        if not val_label is None:
            if self.val_ks_metrics is None:
                tc.sim_cut_by_value(val_label, val_pred)
                val_ks_df = pd.DataFrame(tc.metrics)
                val_bst_threshold = tc.bst_threshold
                val_bst_score = tc.bst_score
            else:
                logger.info(f"Using calculated val ks")
                val_ks_df = pd.DataFrame(self.val_ks_metrics)
                val_bst_threshold = self.val_ks_bst_threshold
                val_bst_score = self.val_ks_bst_score

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
        except:
            lift_file_path = Path(
                self.metric_dir, f"{self.model_info['name']}_plot_lift.json"
            )
        try:
            gain_file_path = Path(
                self.metric_dir, self.train_conf['output']['plot_gain']['name']
            )
        except:
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

        if not val_label is None:
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

    def _write_pr_data(self, train_label, train_pred, val_label=None, val_pred=None, n_cuts=1000) -> None:
        # Set up file path
        try:
            pr_curve_path = Path(
                self.metric_dir,
                self.train_conf['output']['plot_precision_recall']['name']
            )
        except:
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
            logger.info(f"Too much points in training pr curve, pruning")
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
        if not val_label is None:
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
        except:
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
        except:
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
