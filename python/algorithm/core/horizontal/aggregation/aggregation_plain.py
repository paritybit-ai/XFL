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


from typing import List, OrderedDict, Tuple

import torch
import numpy as np

from service.fed_config import FedConfig
from .aggregation_base import AggregationRootBase, AggregationLeafBase


class AggregationPlainLeaf(AggregationLeafBase):
    def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
        # super().__init__(sec_conf, root_id, leaf_ids)
        super().__init__(sec_conf, root_id, FedConfig.node_id)
        self.leaf_ids = leaf_ids or FedConfig.get_label_trainer() + FedConfig.get_trainer()
        
    def _calc_upload_value(self, parameters: OrderedDict, parameters_weight: float) -> Tuple[OrderedDict, float]:
        def f(x):
            y = x[1]
            if isinstance(x[1], torch.Tensor):
                y = x[1].cpu()
            return (x[0], y * parameters_weight)
        
        weighted_parameters = OrderedDict(map(f, parameters.items()))
        
        return (weighted_parameters, parameters_weight)
    

class AggregationPlainRoot(AggregationRootBase):
    def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
        super().__init__(sec_conf, root_id, leaf_ids)
    
    def _calc_aggregated_params(self, received_value: List) -> OrderedDict:
        total_weight = sum([item[1] for item in received_value])
        
        if self.initial_parameters is not None:
            parameters = self.initial_parameters
        else:
            parameters = received_value[0][0]
            
        for k in parameters.keys():
            for item in received_value[1:]:
                received_value[0][0][k] += item[0][k]
            if received_value[0][0][k].dtype in [np.float32, np.float64]:
                received_value[0][0][k] /= total_weight
            elif received_value[0][0][k].dtype not in [torch.float32, torch.float64]:
                ori_dtype = received_value[0][0][k].dtype
                received_value[0][0][k] = received_value[0][0][k].to(dtype=torch.float32)
                received_value[0][0][k] /= total_weight
                received_value[0][0][k] = received_value[0][0][k].to(dtype=ori_dtype)
            else:
                received_value[0][0][k] /= total_weight

        return received_value[0][0]
        

