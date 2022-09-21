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
import math
import pickle
from typing import List, OrderedDict, Tuple, Dict

from common.communication.gRPC.python.channel import DualChannel
from service.fed_config import FedConfig

MAX_BLOCK_SIZE = 524288000 # 500M
MOV = b"@" # middle of value
EOV = b"&" # end of value


class AggregationLeafBase(object):
    __metaclass__ = abc.ABCMeta

    # def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
    def __init__(self, sec_conf: dict, root_id: str = '', leaf_id: str = '') -> None:
        self.sec_conf = sec_conf
        if root_id:
            self.root_id = root_id
        else:
            self.root_id = FedConfig.get_assist_trainer()
        
        if leaf_id:
            self.leaf_id = leaf_id
        else:
            # self.leaf_ids = FedConfig.get_label_trainer() + FedConfig.get_trainer()
            self.leaf_id = FedConfig.node_id
            
        # self.aggregation_chan = BroadcastChannel(name='aggregation', ids=self.leaf_ids + [self.root_id],
        #                                     root_id=self.root_id, auto_offset=True)
        
        self.aggregation_chann = DualChannel(name='aggregation_' + self.leaf_id,
                                             ids=[self.root_id, self.leaf_id])
        
        
        # self.seg_chan = BroadcastChannel(name='num_seg', ids=self.leaf_ids + [self.root_id],
        #                                     root_id=self.root_id, auto_offset=True)
    @abc.abstractmethod
    def _calc_upload_value(self, parameters: OrderedDict, parameters_weight: float) -> Tuple[OrderedDict, float]:
        pass

    def upload(self, parameters: OrderedDict, parameters_weight: float) -> int:
        """ Send (parameters * parameters_weight, parameter_weight) to assist_trainer
        """
        # response_codes = []
        # value = self._calc_upload_value(parameters, parameters_weight)
        # pickle_value = pickle.dumps(value)
        # for seg in max_bytes_segementation(pickle_value):
        #     response_code = self.aggregation_chan.send(seg, use_pickle=False)
        #     response_codes.append(response_code)
        # self.seg_chan.send(len(response_codes))
        # return response_codes

        # value = self._calc_upload_value(parameters, parameters_weight)
        # return self.aggregation_chan.send(value)

        response_codes = []
        value = self._calc_upload_value(parameters, parameters_weight)
        pickle_value = pickle.dumps(value)
        for seg in max_bytes_segementation(pickle_value):
            response_code = self.aggregation_chann.send(seg, use_pickle=False)
            response_codes.append(response_code)
        return int(any(response_codes))
    
    def download(self) -> OrderedDict:
        """ Receive global parameters from assist_trainer
        """
        # pickle_params = b""
        # num_seg = self.seg_chan.recv()
        # for n in range(num_seg):
        #     pickle_params += self.aggregation_chan.recv(use_pickle=False)
        # params = pickle.loads(pickle_params)
        # return params

        # params = self.aggregation_chan.recv()
        # return params
        pickle_params = bytes()
        while True:
            recv_value = self.aggregation_chann.recv(use_pickle=False)
            pickle_params += recv_value[:-1]
            if recv_value[-1] == EOV[0]:
                break
            elif recv_value[-1] == MOV[0]:
                continue    
        params = pickle.loads(pickle_params)
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
            
        self.aggregation_channs: Dict[str, DualChannel] = {}
        for id in self.leaf_ids:
            self.aggregation_channs[id] = DualChannel(name='aggregation_' + id,
                                                      ids=[self.root_id, id])
            
        # self.aggregation_chan = BroadcastChannel(name='aggregation', ids=self.leaf_ids + [self.root_id],
        #                                     root_id=self.root_id, auto_offset=True)
        # self.seg_chan = BroadcastChannel(name='num_seg', ids=self.leaf_ids + [self.root_id],
        #                                     root_id=self.root_id, auto_offset=True)
        
    @abc.abstractmethod
    def _calc_aggregated_params(self, received_value: List) -> OrderedDict:
        pass
    
    def set_initial_params(self, params: OrderedDict) -> None:
        self.initial_parameters = params

    def aggregate(self) -> OrderedDict:
        """ receive local gradient/weights from trainer, then calculate average gradient/weights.
        """
        # received_values = []
        # num_seg = self.seg_chan.collect()
        # for n in range(num_seg[0]):
        #     if n == 0:
        #         received_values = self.aggregation_chan.collect(use_pickle=False)
        #     else:    
        #         for n, value in enumerate(self.aggregation_chan.collect(use_pickle=False)):
        #             received_values[n] += value
        # received_values = list(map(lambda x: pickle.loads(x), received_values))
        # aggregated_params = self._calc_aggregated_params(received_values)
        # return aggregated_params
        received_values = []
        # collect_flg = []
        
        is_continue_flags = [True for party_id in self.aggregation_channs]
        received_values = [bytes() for party_id in self.aggregation_channs]
        
        while True:
            # collect_values = self.aggregation_chan.collect(use_pickle=False)
            for i, id in enumerate(self.leaf_ids):
                if not is_continue_flags[i]:
                    continue
                
                data = self.aggregation_channs[id].recv(use_pickle=False, wait=False)

                if data is None:
                    continue
                
                received_values[i] += data[:-1]
                
                if data[-1] == EOV[0]:
                    received_values[i] = pickle.loads(received_values[i])
                    is_continue_flags[i] = False
                    
            flag = any(is_continue_flags)
            if not flag:
                break
                    
            #     if len(collect_flg) == 0:
            #         collect_flg = [False for _ in range(len(collect_values))]
            #     if len(received_values) == 0:
            #         received_values = [bytes() for _ in range(len(collect_values))]
            #     for n, value in enumerate(collect_values):
            #         if value is None:
            #             continue
            #         received_values[n] += value[:-1]
            #         if value[-1] == EOV[0]:
            #             collect_flg[n] = True
            #         elif value[-1] == MOV[0]:
            #             continue
            # if all(collect_flg):
            #     break

        # received_values = list(map(lambda x: pickle.loads(x), received_values))
        aggregated_params = self._calc_aggregated_params(received_values)
        return aggregated_params

    def broadcast(self, params: OrderedDict) -> int:
        # br_status = []
        # pickle_params = pickle.dumps(params)
        # for seg in max_bytes_segementation(pickle_params):
        #     br_code = self.aggregation_chan.broadcast(seg, use_pickle=False)
        #     br_status.append(br_code)
        # self.seg_chan.broadcast(len(br_status))
        # return br_status
        # return self.aggregation_chan.broadcast(params)
        # br_status = []
        # pickle_params = pickle.dumps(params)
        # for seg in max_bytes_segementation(pickle_params):
        #     br_code = self.aggregation_chan.broadcast(seg, use_pickle=False)
        #     br_status.append(br_code)
        
        br_status = []
        pickle_params = pickle.dumps(params)
        for seg in max_bytes_segementation(pickle_params):
            br_codes = []
            for id in self.leaf_ids:
                br_code = self.aggregation_channs[id].send(seg, use_pickle=False)
                br_codes.append(br_code)
            br_status.append(any(br_codes))
        return int(any(br_status))
        

def max_bytes_segementation(value):
    # n = math.ceil(1.0 * len(value) / MAX_BLOCK_SIZE)
    # for i in range(n):
    #     max_segement = value[i*MAX_BLOCK_SIZE: (i+1)*MAX_BLOCK_SIZE]
    #     yield max_segement
    n = math.ceil(1.0 * len(value) / MAX_BLOCK_SIZE)
    for i in range(n):
        if i == n-1:
            max_segement = value[i*MAX_BLOCK_SIZE: (i+1)*MAX_BLOCK_SIZE] + EOV
        else:    
            max_segement = value[i*MAX_BLOCK_SIZE: (i+1)*MAX_BLOCK_SIZE] + MOV
        yield max_segement