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

import numpy as np
import pandas as pd

from algorithm.core.data_io import CsvReader, ValidationNumpyDataset
from algorithm.core.encryption_param import get_encryption_param
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.framework.vertical.vertical_model_base import VerticalModelBase
from common.utils.logger import logger


class VerticalXgboostBase(VerticalModelBase):
    def __init__(self, train_conf: dict, is_label_trainer: bool = False, *args, **kwargs):
        super().__init__(train_conf)
        self.train_conf = train_conf
        self.train_features, self.train_label = None, None
        self.val_features, self.val_label = None, None
        self.xgb_config = None
        self.is_label_trainer = is_label_trainer
        self.feature_importances_ = {}
        self.__init_xgb_config()
        self.__init_data()
        self.__convert_to_binned_data()

    def __init_data(self) -> None:
        """ Init data, include features and label.

        Returns: None

        """
        self.bs = self.train_params.get("validation_batch_size")

        if self.input_trainset:
            _ = self.__load_data(self.input_trainset)
            self.train_features, self.train_label, self.train_ids, self.train_dataset = _

        if self.input_valset:
            _ = self.__load_data(self.input_valset)
            self.val_features, self.val_label, self.val_ids, self.val_dataset = _

        if self.input_testset:
            _ = self.__load_data(self.input_testset)
            self.test_features, self.test_label, self.test_ids, self.test_dataset = _

    def __convert_to_binned_data(self):
        if not self.input_trainset:
            return
        out = pd.Series(self.train_features.columns).apply(
            lambda x: pd.cut(self.train_features[x], bins=self.xgb_config.num_bins,
                             retbins=True, labels=range(self.xgb_config.num_bins))
        )

        if self.xgb_config.num_bins <= 256:
            dtype = np.uint8
        elif self.xgb_config.num_bins <= 2 ** 16:
            dtype = np.uint16
        else:
            dtype = np.uint32

        self.train_features = pd.DataFrame([out[i][0] for i in range(len(out))], dtype=dtype).T
        self.split_points = [out[i][1][1:-1] for i in range(len(out))]

    def __load_data(self, config):
        """ Load data from dataset config.

        Args:
            argv: Dataset config.

        Returns: [CsvReader, ...]

        """
        if len(config) > 1:
            logger.warning("More than one dataset is not supported.")

        config = config[0]
        if config["type"] == "csv":
            data_reader = CsvReader(path=os.path.join(config["path"], config["name"]),
                                    has_id=config["has_id"], has_label=config["has_label"])
            features = data_reader.features(type="pandas.dataframe")
            features.replace({np.nan: 0, self.xgb_config.missing_value: 0}, inplace=True)
            ids = data_reader.ids
            if self.is_label_trainer:
                labels = data_reader.label()
            else:
                labels = None
            dataset = ValidationNumpyDataset(
                batch_size=self.bs,
                dataset=features.to_numpy().astype(np.float32),
                label=labels
            )
            feature_names = data_reader.feature_names()
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return features, labels, ids, dataset

    def __init_xgb_config(self) -> None:
        """ Init xgboost config.

        Returns: None

        """
        default_config = self.train_info.get("params")
        encryption_methods = list(default_config.get("encryption_params", {}).keys())
        if len(encryption_methods) > 0:
            encryption_method = encryption_methods[0]
        else:
            encryption_method = "plain"
        encryption_param = default_config.get("encryption_params", {"plain": {}})[encryption_method]

        self.xgb_config = XGBTreeParam(task_type=default_config.get("task_type"),
                                       loss_param=default_config.get("lossfunc_config"),  # ("BCEWithLogits"),
                                       num_trees=default_config.get("num_trees"),
                                       learning_rate=default_config.get("learning_rate"),
                                       gamma=default_config.get("gamma"),
                                       lambda_=default_config.get("lambda_"),
                                       max_depth=default_config.get("max_depth"),
                                       num_bins=default_config.get("num_bins", 16),
                                       min_split_gain=default_config.get("min_split_gain"),
                                       min_sample_split=default_config.get("min_sample_split"),
                                       min_leaf_node=default_config.get("min_leaf_node"),
                                       feature_importance_type=default_config.get("feature_importance_type"),
                                       run_goss=default_config.get("run_goss", False),
                                       top_rate=default_config.get("top_rate"),
                                       other_rate=default_config.get("other_rate"),
                                       validation_freqs=1,
                                       metrics=default_config.get("metric_config"),
                                       early_stopping_param=default_config.get("early_stopping_params",
                                                                               {"patience": -1,
                                                                                "key": "ks",
                                                                                "delta": 0.001}),
                                       encryption_param=get_encryption_param(encryption_method, encryption_param),
                                       subsample_feature_rate=default_config.get("subsample_feature_rate"),
                                       missing_value=float('inf'),
                                       max_num_cores=default_config.get("max_num_cores", 999),
                                       col_batch=default_config.get("col_batch"),
                                       row_batch=default_config.get("row_batch"))
