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
        train_reader: CsvReader = self.__load_data(self.input_trainset)
        validation_reader: CsvReader = self.__load_data(self.input_valset)
        if self.is_label_trainer:
            self.train_features,  self.train_label = train_reader.features(type="pandas.dataframe"), train_reader.label()
            self.val_features, self.val_label = validation_reader.features(type="pandas.dataframe"), validation_reader.label()
            self.train_feature_name, self.valid_feature_name = train_reader.feature_names(), validation_reader.feature_names()
        else:
            self.train_features = train_reader.features(type="pandas.dataframe")
            self.val_features = validation_reader.features(type="pandas.dataframe")

        self.train_ids = train_reader.ids
        self.val_ids = validation_reader.ids

        self.train_features.replace({np.nan: 0, self.xgb_config.missing_value: 0}, inplace=True)
        self.val_features.replace({np.nan: 0, self.xgb_config.missing_value: 0}, inplace=True)
        
        self.bs = self.train_params.get("validation_batch_size")
        self.train_dataset = ValidationNumpyDataset(batch_size=self.bs, dataset=self.train_features.to_numpy().astype(np.float32),
                                                    label=self.train_label)
        self.val_dataset = ValidationNumpyDataset(batch_size=self.bs, dataset=self.val_features.to_numpy().astype(np.float32), 
                                                  label=self.val_label)
        
    def __convert_to_binned_data(self):
        out = pd.Series(self.train_features.columns).apply(
            lambda x: pd.cut(self.train_features[x], bins=self.xgb_config.num_bins, retbins=True, labels=range(self.xgb_config.num_bins))
        )
        
        if self.xgb_config.num_bins <= 256:
            dtype = np.uint8
        elif self.xgb_config.num_bins <= 1e16:
            dtype = np.uint16
        else:
            dtype = np.uint32
            
        self.train_features = pd.DataFrame([out[i][0] for i in range(len(out))], dtype=dtype).T
        self.split_points = [out[i][1][1:-1] for i in range(len(out))]
        
    def __load_data(self, config) -> CsvReader:
        """ Load data from dataset config.

        Args:
            argv: Dataset config.

        Returns: CsvReader.

        """
        if len(config) > 1:
            logger.warning("More than one dataset is not supported.")
            
        config = config[0]
        if config["type"] == "csv":
            data_reader = CsvReader(path=os.path.join(config["path"],config["name"]), has_id=config["has_id"], has_label=config["has_label"])
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return data_reader

    def __init_xgb_config(self) -> None:
        """ Init xgboost config.

        Returns: None

        """
        default_config = self.train_info.get("params")
        encryption_method = list(default_config.get("encryption_params").keys())[0]
        encryption_param = default_config.get("encryption_params")[encryption_method]

        self.xgb_config = XGBTreeParam(task_type=default_config.get("task_type"),
                                       loss_param=default_config.get("lossfunc_config"), #("BCEWithLogits"),
                                       num_trees=default_config.get("num_trees"),
                                       learning_rate=default_config.get("learning_rate"),
                                       gamma=default_config.get("gamma"),
                                       lambda_=default_config.get("lambda_"),
                                       max_depth=default_config.get("max_depth"),
                                       num_bins=default_config.get("num_bins"),
                                       min_split_gain=default_config.get("min_split_gain"),
                                       min_sample_split=default_config.get("min_sample_split"),
                                       min_leaf_node=default_config.get("min_leaf_node"),
                                       feature_importance_type=default_config.get("feature_importance_type"),
                                       run_goss=default_config.get("run_goss"),
                                       top_rate=default_config.get("top_rate"),
                                       other_rate=default_config.get("other_rate"),
                                       validation_freqs=1,
                                       metrics=default_config.get("metric_config"),
                                       early_stopping_param=default_config.get("early_stopping_params"),
                                       encryption_param=get_encryption_param(encryption_method, encryption_param),
                                       subsample_feature_rate=default_config.get("subsample_feature_rate"),
                                       missing_value=float('inf'),
                                       max_num_cores=default_config.get("max_num_cores"),
                                       col_batch=default_config.get("col_batch"),
                                       row_batch=default_config.get("row_batch"))
