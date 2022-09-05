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
from typing import Any

import numpy as np
import pandas as pd

from algorithm.core.data_io import CsvReader, NdarrayIterator
from algorithm.core.encryption_param import get_encryption_param
from algorithm.core.tree.cat_param_parser import parse_category_param
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.framework.vertical.vertical_model_base import VerticalModelBase
from common.utils.logger import logger


class VerticalXgboostBase(VerticalModelBase):
    def __init__(self, train_conf: dict, is_label_trainer: bool = False, *args, **kwargs):
        super().__init__(train_conf)
        self.train_conf = train_conf
        self.train_features, self.train_label, self.train_ids = None, None, None
        self.val_features, self.val_label, self.val_ids = None, None, None
        self.test_features, self.test_label, self.test_ids = None, None, None
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
            self.train_features, self.train_label, self.train_ids = _
            self.train_dataset = NdarrayIterator(self.train_features.to_numpy(), self.bs)
        else:
            self.train_dataset = None

        if self.input_valset:
            _ = self.__load_data(self.input_valset)
            self.val_features, self.val_label, self.val_ids = _
            self.val_dataset = NdarrayIterator(self.val_features.to_numpy(), self.bs)
        else:
            self.val_dataset = None

        if self.input_testset:
            _ = self.__load_data(self.input_testset)
            self.test_features, self.test_label, self.test_ids = _
            self.test_dataset = NdarrayIterator(self.test_features.to_numpy(), self.bs)
        else:
            self.test_dataset = None

    def __convert_to_binned_data(self):
        ''' Note self.train_features will be converted to binned feature '''
        cat_columns = parse_category_param(self.train_features,
                                           col_index=self.xgb_config.cat_col_index,
                                           col_names=self.xgb_config.cat_col_names,
                                           col_index_type=self.xgb_config.cat_col_index_type,
                                           col_names_type=self.xgb_config.cat_col_names_type,
                                           max_num_value=self.xgb_config.cat_max_num_value,
                                           max_num_value_type=self.xgb_config.cat_max_num_value_type)
        self.cat_columns = cat_columns
        self.cat_feature_names = []
        
        if len(cat_columns) > 0:
            self.cat_feature_names = self.train_features.columns[cat_columns].to_list()
            self.train_features[self.cat_feature_names] = self.train_features[self.cat_feature_names].astype('category')
            
        def f(x):
            if self.train_features[x].dtypes == "category":
                value_counts = self.train_features[x].value_counts()  # descending order
                
                if value_counts.shape[0] > self.xgb_config.num_bins:
                    values = value_counts.index.to_list()
                    list_unique = values[:self.xgb_config.num_bins - 1]
                    list_group = values[self.xgb_config.num_bins - 1:]
                    uniques = np.array(list_unique + [list_group], dtype=np.object)
                    value_map = {v: i for i, v in enumerate(list_unique)}
                    value_map.update({v: len(list_unique) for v in list_group})
                    codes = self.train_features[x].map(value_map)
                else:
                    codes, uniques = pd.factorize(self.train_features[x], na_sentinel=0)  # na_sentinel will not be activated actually
                    uniques = uniques.to_numpy()
                    
                # uniques: array of values that belongs to the same category
                # codes: binned values
                return pd.Series(codes, name=x), uniques.tolist()
            else:
                binned_values, split_points = pd.cut(self.train_features[x], bins=self.xgb_config.num_bins, retbins=True, labels=range(self.xgb_config.num_bins))
                return binned_values, split_points

        if self.input_trainset:
            out = pd.Series(self.train_features.columns).apply(f)
            
            if self.xgb_config.num_bins <= 256:
                dtype = np.uint8
            elif self.xgb_config.num_bins <= 2 ** 16:
                dtype = np.uint16
            else:
                dtype = np.uint32
                
            self.train_features = pd.DataFrame([out[i][0] for i in range(len(out))], dtype=dtype).T
            
            # For continuous features, self.split_points stores the split points between bins, for example, 15 split points for 16 bins.
            # For categorial features, self.split_points stores original values correspond to the bin values, for example, 16 values for 16 bins.
            self.split_points = [out[i][1][1:-1] if i not in self.cat_columns else out[i][1][:] for i in range(len(out))]
        
    def __load_data(self, config):
        """ Load data from dataset config.

        Args:
            config: Dataset config.

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
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return features, labels, ids
    
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
            
    def __init_xgb_config(self) -> None:
        """ Init xgboost config.

        Returns: None

        """
        default_config = self.train_info.get("params")
        cat_params = default_config.get("cat_feature", {})
        encryption_methods = list(default_config.get("encryption_params", {}).keys())
        if len(encryption_methods) > 0:
            encryption_method = encryption_methods[0]
        else:
            encryption_method = "plain"
        encryption_param = default_config.get("encryption_params", {"plain": {}})[encryption_method]

        self.xgb_config = XGBTreeParam(task_type=default_config.get("task_type"),
                                       loss_param=default_config.get("lossfunc_config"),  # ("BCEWithLogitsLoss"),
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
                                       missing_value=float('inf'),  # TODO: check
                                       max_num_cores=default_config.get("max_num_cores", 999),
                                       col_batch=default_config.get("col_batch", 64),
                                       row_batch=default_config.get("row_batch", 40000),
                                       cat_col_index=cat_params.get("col_index", ""),
                                       cat_col_names=cat_params.get("col_names", []),
                                       cat_max_num_value=cat_params.get("max_num_value", 0),
                                       cat_col_index_type=cat_params.get("col_index_type", "inclusive"),
                                       cat_col_names_type=cat_params.get("col_names_type", "inclusive"),
                                       cat_max_num_value_type=cat_params.get("max_num_value_type", "union"),
                                       cat_smooth=default_config.get("cat_smooth", 0))

