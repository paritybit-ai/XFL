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
import math
import sys

import numpy as np
from sklearn import metrics as sklearn_metrics

from common.xregister import xregister

metric_dict = {
    "accuracy": "accuracy_score",
    "acc": "accuracy_score",
    "precision": "precision_score",
    "recall": "recall_score",
    "auc": "roc_auc_score",
    "mape": "mean_absolute_percentage_error",
    "mse": "mean_squared_error",
    "mae": "mean_absolute_error",
    "r2": "r2_score",
    "median_ae": "median_absolute_error",
    "rmse": "root_mean_squared_error"
}


def get_metric(name: str):
    if name in metric_dict.keys():
        name = metric_dict[name]
    else:
        name = name
    if name in dir(sklearn_metrics):
        metric = getattr(sklearn_metrics, name)
    elif name in dir(sys.modules[__name__]):
        metric = getattr(sys.modules[__name__], name)
    elif name in xregister.registered_object:
        metric = xregister(name)
    else:
        raise ValueError(f"Metric {name} is not defined.")
    return metric


def ks(y_true, y_pred):
    fpr, tpr, _ = sklearn_metrics.roc_curve(y_true, y_pred)
    ks = max(np.max(tpr - fpr), 0)
    return ks


def root_mean_squared_error(y_true, y_pred):
    mse_value = sklearn_metrics.mean_squared_error(y_true, y_pred)
    rmse_value = math.sqrt(mse_value)
    return rmse_value
