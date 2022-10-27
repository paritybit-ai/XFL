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

from typing import OrderedDict
from algorithm.core.horizontal.aggregation.aggregation_base import AggregationLeafBase
from .base import BaseTrainer
from collections import OrderedDict

class FedAvgLabelTrainer(BaseTrainer):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        
        self.register_hook(place="before_local_epoch", rank=1,
                           func=self._download_model, desc="download global model")
        self.register_hook(place="after_local_epoch", rank=1,
                           func=self._upload_model, desc="upload local model")
        
    def _download_model(self, context: dict):
        aggregator: AggregationLeafBase = self.aggregator
        new_state_dict = aggregator.download()
        self.model.set_state_dict(new_state_dict)
        
    def _upload_model(self, context: dict):
        aggregator: AggregationLeafBase = self.aggregator
        state_dict = self._rebuild_state_dict(self.model.state_dict())
        aggregation_config = self.train_params["aggregation_config"]
        weight = aggregation_config.get("weight") or len(self.train_dataloader.dataset)
        aggregator.upload(state_dict, weight)

    def _rebuild_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v.numpy()
        return new_state_dict