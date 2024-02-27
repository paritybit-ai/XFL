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


from pathlib import Path

from common.utils.config_sync import ConfigSynchronizer
from common.checker.x_types import All
from common.utils.config_parser import CommonConfigParser
from algorithm.core.tree.tree_param import XGBTreeParam
from algorithm.core.encryption_param import get_encryption_param


class Common:
    def __init__(self, train_conf: dict):
        sync_rule = {
            "train_info": {
                "interaction_params": All(),
                "train_params": {
                    "encryption": All(),
                    "num_trees": All(),
                    "learning_rate": All(),
                    "gamma": All(),
                    "lambda_": All(),
                    "max_depth": All(),
                    "num_bins": All(),
                    "min_split_gain": All(),
                    "min_sample_split": All(),
                    "feature_importance_type": All(),
                    "lossfunc": All(),
                    "metric": All(),
                    "early_stopping": All()
                }
            }
        }
        train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
        self.train_conf = train_conf
        self.model_info = train_conf.get("model_info")
        self.common_config = CommonConfigParser(train_conf)
        self.__init_xgb_config()

    def __init_xgb_config(self):
        default_config = self.common_config.train_params
        self.save_dir = Path(self.common_config.output.get("path", ""))
        self.metric_dir = self.common_config.output.get("path", "")
        self.save_frequency=self.common_config.save_frequency
        self.echo_training_metrics = self.common_config.echo_training_metrics
        self.write_training_prediction = self.common_config.write_training_prediction
        self.write_validation_prediction = self.common_config.write_validation_prediction
        cat_params = default_config.get("category", {}).get("cat_features", {})
        encryption_method = list(default_config.get("encryption", {}).keys())
        if len(encryption_method) > 0:
            self.encryption_method = encryption_method[0]
        else:
            self.encryption_method = "plain"
        self.encryption_params = self.common_config.encryption
        downsampling_params = default_config.get("downsampling", {})
        self.xgb_config = XGBTreeParam(
            loss_param=self.common_config.lossfunc,  # ("BCEWithLogitsLoss"),
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
            run_goss=downsampling_params.get("row", {}).get("run_goss", False),
            top_rate=downsampling_params.get("row", {}).get("top_rate"),
            other_rate=downsampling_params.get("row", {}).get("other_rate"),
            metrics=self.common_config.metric,
            early_stopping_param=self.common_config.early_stopping,
            encryption_param=get_encryption_param(
                self.encryption_method, 
                self.encryption_params[self.encryption_method]
            ),
            subsample_feature_rate=downsampling_params.get("column", {}).get("rate", 1.0),
            missing_value=float('inf'),
            max_num_cores=default_config.get("max_num_cores", 999),
            col_batch=default_config.get("advanced", {}).get("col_batch", 64),
            row_batch=default_config.get("advanced", {}).get("row_batch", 40000),
            cat_col_index=cat_params.get("col_index", ""),
            cat_col_names=cat_params.get("col_names", []),
            cat_max_num_value=cat_params.get("max_num_value", 0),
            cat_col_index_type=cat_params.get("col_index_type", "inclusive"),
            cat_col_names_type=cat_params.get("col_names_type", "inclusive"),
            cat_max_num_value_type=cat_params.get("max_num_value_type", "union"),
            cat_smooth=default_config.get("category", {}).get("cat_smooth", 1.0)
        )