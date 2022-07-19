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


import numpy as np

from algorithm.core.metrics import get_metric, ks


def test_get_metric():
    y_true=np.array([1,1,0,0])
    y_pred=np.array([1,0,1,0])
    metric = get_metric('acc')
    assert metric(y_true,y_pred) == 0.5

    metric = get_metric('auc')
    assert metric(y_true,y_pred) == 0.5

    metric = get_metric('recall')
    assert metric(y_true,y_pred) == 0.5

def test_ks():
    y_true=np.array([1,1,0,0])
    y_pred=np.array([0.8,0.5,0.1,0.1])
    ks_value = ks(y_true, y_pred)
    assert ks_value == 1.0
