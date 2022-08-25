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


from typing import Optional, List

from algorithm.core.encryption_param import EncryptionParam


class EarlyStoppingParam(object):
    pass


class LossParam(object):
    def __init__(self, name):
        self.name = name


class XGBTreeParam(object):
    def __init__(self,
                 task_type: str,  # 'classification', 'regression'
                 loss_param: LossParam,  # ["cross_entropy", "lse", "lae", "huber", "fair", "log_cosh", "tweedie"]
                 num_trees: int,
                 learning_rate: float,
                 gamma: float,
                 lambda_: float,
               
                 max_depth: int,
                 num_bins: int,
                 min_split_gain: float,
                 min_sample_split: int,
                 min_leaf_node: int,
                 feature_importance_type: str,
               
                 run_goss: bool,
                 top_rate: float,
                 other_rate: float,
               
                 validation_freqs: int,
                 metrics: List[str],
               
                 early_stopping_param: Optional[EarlyStoppingParam] = None,  # 'split',(split time) 'gain'(split gain)
                 encryption_param: Optional[EncryptionParam] = None,
                 subsample_feature_rate: float = 1.0,
                 missing_value: float = float('inf'),
                 max_num_cores: int = 9999,
                 
                 col_batch: int = 128,
                 row_batch: int = 10000,
                 
                 # category feature params
                 cat_col_index: str = "",
                 cat_col_names: List[str] = [],
                 cat_max_num_value: int = 0,
                 cat_col_index_type: str = 'inclusive',
                 cat_col_names_type: str = 'inclusive',
                 cat_max_num_value_type: str = 'union',
                 
                 cat_smooth: float = 0):

        self.task_type = task_type
        self.loss_param = loss_param
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.early_stopping_param = early_stopping_param
        self.encryption_param = encryption_param
        
        # single tree training
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.min_split_gain = min_split_gain
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.feature_importance_type = feature_importance_type
        self.subsample_feature_rate = subsample_feature_rate
        self.missing_value = missing_value
        
        self.run_goss = run_goss
        self.top_rate = top_rate
        self.other_rate = other_rate

        # validation
        self.validation_freqs = validation_freqs
        self.metrics = metrics
        
        # multiprocess
        self.max_num_cores = max_num_cores
        
        self.col_batch = col_batch
        self.row_batch = row_batch
        
        self.cat_col_index = cat_col_index
        self.cat_col_names = cat_col_names
        self.cat_max_num_value = cat_max_num_value
        self.cat_col_index_type = cat_col_index_type
        self.cat_col_names_type = cat_col_names_type
        self.cat_max_num_value_type = cat_max_num_value_type
        
        self.cat_smooth = cat_smooth
