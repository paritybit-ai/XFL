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


import inspect
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.nn import BCELoss

from common.utils.algo_utils import (BiClsAccuracy, BiClsAuc, BiClsF1, BiClsKS,
                                     BiClsPrecision, BiClsRecall)
from common.utils.logger import logger


class BiClsMetric:

    def __init__(self, epoch, output_file=None, metric_config={}, lossfunc_conifg={}):
        self.metric_functions_map = {
            "BCEWithLogitsLoss": BCELoss,
            "acc": BiClsAccuracy,
            "precision": BiClsPrecision,
            "recall": BiClsRecall,
            "f1_score": BiClsF1,
            "auc": BiClsAuc,
            "ks": BiClsKS
        }
        self.metric_functions = {}
        self.metrics = {}
        self.epoch = epoch
        self.output_file = output_file

        loss_function = lossfunc_conifg.get("method")
        if loss_function:
            if loss_function not in self.metric_functions_map:
                raise NotImplementedError("Loss function {} is not supported in this model.".format(loss_function))
            func = self.metric_functions_map[loss_function]
            method_args = inspect.getfullargspec(func).args
            defined_args = {}
            for (key, value) in lossfunc_conifg.items():
                if key in method_args:
                    defined_args[key] = value
            self.metric_functions[loss_function] = func(**defined_args)

        for metric_function in metric_config:
            if metric_function == "auc_ks":
                logger.warning('metric "auc_ks" in config will be deprecated in future version, '
                               'please use "auc" and "ks" separately.')
                defined_args = {}
                for _ in ["auc", "ks"]:
                    func = self.metric_functions_map[_]
                    self.metric_functions[_] = func(**defined_args)
                continue
            if metric_function == "decision_table":
                continue
            elif metric_function not in self.metric_functions_map:
                raise NotImplementedError("Metric function {} is not supported in this model.".format(metric_function))
            func = self.metric_functions_map[metric_function]
            defined_args = {}
            self.metric_functions[metric_function] = func(**defined_args)

    def calc_metrics(self, y_true: np.array, y_pred: np.array):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred > 0.5)
        for metric_function in self.metric_functions:
            if metric_function in ("acc", "precision", "recall", "f1_score"):
                self.metrics[metric_function] = self.metric_functions[metric_function](cm).item()
            elif metric_function in ("auc", "ks"):
                self.metrics[metric_function] = self.metric_functions[metric_function](tpr, fpr).item()
            elif metric_function == "BCEWithLogitsLoss":
                self.metrics[metric_function] = self.metric_functions[metric_function](torch.tensor(y_pred),
                                                                                       torch.tensor(y_true)).item()

    def __repr__(self):
        output = ["epoch: %d" % self.epoch]
        for k, v in self.metrics.items():
            output.append("%s: %.6g" % (k, v))
        return ', '.join(output)

    def save(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'a') as f:
                features = []
                for k in self.metric_functions_map:
                    if k in self.metrics:
                        features.append("%.6g" % self.metrics[k])
                    else:
                        features.append("")
                f.write("%d,%s\n" % (self.epoch, ','.join(features)))
        else:
            with open(self.output_file, 'w') as f:
                f.write("%s,%s\n" % ("epoch", ','.join([_ for _ in self.metric_functions_map])))
                features = []
                for k in self.metric_functions_map:
                    if k in self.metrics:
                        features.append("%.6g" % self.metrics[k])
                    else:
                        features.append("")
                f.write("%d,%s\n" % (self.epoch, ','.join(features)))


class ThresholdCutter:
    def __init__(self, output_file=None):
        self.bst_threshold = 0.5
        self.bst_score = 0
        # self.default_threshold = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
        #                           0.51, 0.52, 0.53, 0.54, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
        self.default_percentile = np.arange(100, -1, -1)
        self.output_file = output_file
        self.metrics = {
            "threshold": [],
            "tn": [],
            "fp": [],
            "fn": [],
            "tp": [],
            "tpr": [],
            "fpr": [],
            "ks": []
        }

    def cut_by_value(self, y_true: np.array, y_pred: np.array, values: List = None):
        probs = np.unique(y_pred)
        logger.info("num of probs: %d." % len(probs))
        if values is None:
            if len(probs) < len(self.default_percentile):
                # logger.warning("ks points %d less than the default num: %d." % (len(probs),
                #                                                                 len(self.default_percentile)))
                values = np.array(sorted(probs, reverse=True))
                self.default_percentile = np.array([sum(y_pred < _) / (len(y_pred) - 1) * 100 for _ in values])
            else:
                values = np.percentile(y_pred, self.default_percentile)
        # -	Threshold, TP, FN, FP, TN, TPR, FPR, KS
        for threshold in values:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= threshold).ravel()
            if tp + fn > 0:
                tpr = tp / (tp + fn)
            else:
                tpr = np.nan
            if tn + fp > 0:
                fpr = fp / (tn + fp)
            else:
                fpr = np.nan
            ks = max(np.max(tpr - fpr), 0)
            for metric in self.metrics:
                self.metrics[metric].append(locals()[metric])
            if ks > self.bst_score:
                self.bst_score = ks
                self.bst_threshold = threshold

    def cut_by_index(self, y_true: np.array, y_pred: np.array):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks_curve = tpr - fpr
        ks_curve = np.where(ks_curve > 0, ks_curve, 0)
        idx = np.argmax(ks_curve)
        ks = ks_curve[idx]
        if ks > self.bst_score:
            self.bst_score = ks
            self.bst_threshold = thresholds[idx]

    def save(self):
        df = pd.DataFrame(self.metrics)
        df["top_percentile"] = 100 - self.default_percentile
        df.to_csv(self.output_file, header=True, index=False, float_format='%.6g')


class DecisionTable:
    def __init__(self, conf):
        self.method = conf.get("method", "equal_frequency")
        self.bin_number = conf.get("bin_number", 5)
        self.type = conf.get("type")
        self._check_params()
        self.stats = pd.DataFrame()

    def _check_params(self):
        if self.method not in ("equal_frequency", "equal_width"):
            raise NotImplementedError("decision table: method '{}' is not implemented.".format(self.method))
        if self.bin_number <= 1:
            raise ValueError("decision table: bin number ({}) must be greater than 1.".format(self.bin_number))

    def fit(self, y_true: np.array, y_pred: np.array):
        df = pd.DataFrame({"label": y_true, "pred": y_pred})
        if self.method == "equal_frequency":
            groups = pd.qcut(y_pred, self.bin_number, duplicates='drop', precision=3)
        elif self.method == "equal_width":
            groups = pd.cut(y_pred, self.bin_number, right=True, duplicates='drop', precision=3)
        if self.type == "score_card":
            groups = [pd.Interval(int(_.left), int(_.right), _.closed) for _ in groups]
        df["区间"] = groups
        self.stats["样本数"] = df.groupby("区间").size()
        self.stats["负样本数"] = df.groupby("区间")["label"].agg(lambda x: sum(x == 0))
        self.stats["正样本数"] = df.groupby("区间")["label"].agg(lambda x: sum(x == 1))
        self.stats["区间内负样本占比"] = self.stats["负样本数"] / self.stats["样本数"]
        self.stats["区间内正样本占比"] = self.stats["正样本数"] / self.stats["样本数"]
        self.stats["样本占比"] = self.stats["样本数"] / self.stats["样本数"].sum()
        self.stats["负样本占比"] = self.stats["负样本数"] / self.stats["负样本数"].sum()
        self.stats["正样本占比"] = self.stats["正样本数"] / self.stats["正样本数"].sum()
        self.stats["累计总样本数"] = self.stats["样本数"].cumsum()
        self.stats["累计负样本数"] = self.stats["负样本数"].cumsum()
        self.stats["累计正样本数"] = self.stats["正样本数"].cumsum()
        self.stats["累计负样本/负样本总数"] = self.stats["累计负样本数"] / self.stats["负样本数"].sum()
        self.stats["累计正样本/正样本总数"] = self.stats["累计正样本数"] / self.stats["正样本数"].sum()
        self.stats["累计负样本/累计总样本"] = self.stats["累计负样本数"] / self.stats["累计总样本数"]
        self.stats["累计正样本/累计总样本"] = self.stats["累计正样本数"] / self.stats["累计总样本数"]
        self.stats["累计样本数/总样本数"] = self.stats["累计总样本数"] / self.stats["样本数"].sum()
        self.stats["累计正样本占比/累计总样本占比"] = self.stats["正样本占比"].cumsum() / self.stats["样本占比"].cumsum()
        for _ in ["区间内负样本占比", "区间内正样本占比", "样本占比", "负样本占比", "正样本占比",
                  "累计负样本/负样本总数", "累计正样本/正样本总数", "累计负样本/累计总样本", "累计正样本/累计总样本",
                  "累计样本数/总样本数", "累计正样本占比/累计总样本占比"]:
            self.stats[_] = np.where(
                self.stats[_].isnull(),
                np.nan,
                self.stats[_].apply(lambda x: "%.2f%%" % (x * 100))
            )
        self.stats = self.stats.reset_index()
        self.stats["区间"] = self.stats["区间"].apply(str)

    def save(self, file_name):
        logger.info("decision table: saved to {}.".format(file_name))
        self.stats.to_csv(file_name, header=True, index=False, float_format='%.2g')


class ClusteringMetric:
    def __init__(self):
        pass

    @staticmethod
    def calc_dbi(dist_table, cluster_dist):
        if len(dist_table) == 1:
            return np.nan
        max_dij_list = []
        d = 0
        n = 0
        for i in range(0, len(dist_table)):
            dij_list = []
            for j in range(0, len(dist_table)):
                if j != i:
                    dij_list.append((dist_table[i] + dist_table[j]) / (cluster_dist[d] ** 0.5))
                    d += 1
            dij_list = [_ for _ in dij_list if ~torch.isnan(_)]
            if len(dij_list) <= 0:
                max_dij_list.append(np.nan)
            else:
                max_dij = max(dij_list)
                max_dij_list.append(max_dij)
                n += 1
        if n > 0:
            return np.nansum(max_dij_list) / n
        else:
            return np.nan
