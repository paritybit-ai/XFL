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


import abc
from typing import List, OrderedDict, Tuple

from common.communication.gRPC.python.channel import BroadcastChannel
from service.fed_config import FedConfig


class AggregationLeafBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
        self.sec_conf = sec_conf
        if root_id:
            self.root_id = root_id
        else:
            self.root_id = FedConfig.get_assist_trainer()
        
        if leaf_ids:
            self.leaf_ids = leaf_ids
        else:
            self.leaf_ids = FedConfig.get_label_trainer() + FedConfig.get_trainer()
            
        self.params_chan = BroadcastChannel(name='aggregation', ids=self.leaf_ids + [self.root_id],
                                            root_id=self.root_id, auto_offset=True)
        
    @abc.abstractmethod
    def _calc_upload_value(self, parameters: OrderedDict, parameters_weight: float) -> Tuple[OrderedDict, float]:
        pass

    def upload(self, parameters: OrderedDict, parameters_weight: float) -> int:
        """ Send (parameters * parameters_weight, parameter_weight) to assist_trainer
        """
        value = self._calc_upload_value(parameters, parameters_weight)
        return self.params_chan.send(value)
    
    def download(self) -> OrderedDict:
        """ Receive global parameters from assist_trainer
        """
        params = self.params_chan.recv()
        return params


class AggregationRootBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
        self.sec_conf = sec_conf
        self.initial_parameters = None
        
        if root_id:
            self.root_id = root_id
        else:
            self.root_id = FedConfig.get_assist_trainer()
        
        if leaf_ids:
            self.leaf_ids = leaf_ids
        else:
            self.leaf_ids = FedConfig.get_label_trainer() + FedConfig.get_trainer()
            
        self.params_chan = BroadcastChannel(name='aggregation', ids=self.leaf_ids + [self.root_id],
                                            root_id=self.root_id, auto_offset=True)
        
    @abc.abstractmethod
    def _calc_aggregated_params(self, received_value: List) -> OrderedDict:
        pass
    
    def set_initial_params(self, params: OrderedDict) -> None:
        self.initial_parameters = params

    def aggregate(self) -> OrderedDict:
        """ receive local gradient/weights from trainer, then calculate average gradient/weights.
        """
        received_value = self.params_chan.collect()
        aggregated_params = self._calc_aggregated_params(received_value)
        return aggregated_params
    
    def broadcast(self, params: OrderedDict) -> int:
        return self.params_chan.broadcast(params)
