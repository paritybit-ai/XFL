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


from algorithm.core.horizontal.aggregation.aggregation_base import AggregationRootBase
from ..base import BaseTrainer


class FedProxAssistTrainer(BaseTrainer):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        
        self.register_hook(place="before_local_epoch", rank=1,
                           func=self._broadcast_model, desc="broadcast global model")
        self.register_hook(place="after_local_epoch", rank=1,
                           func=self._aggregate_model, desc="aggregate local models")
        
    def _broadcast_model(self, context: dict):
        aggregator: AggregationRootBase = self.aggregator
        aggregator.broadcast(self.model.state_dict())
        
    def _aggregate_model(self, context: dict):
        aggregator: AggregationRootBase = self.aggregator
        new_state_dict = aggregator.aggregate()
        if self.device != "cpu":
            self._state_dict_to_device(new_state_dict, self.device, inline=True)
        self.model.load_state_dict(new_state_dict)