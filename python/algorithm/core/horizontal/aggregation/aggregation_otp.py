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


import threading
from functools import reduce
from itertools import combinations
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch

from common.communication.gRPC.python.commu import Commu
from common.crypto.csprng.drbg import get_drbg_inst
from common.crypto.csprng.drbg_base import DRBGBase
from common.crypto.key_agreement.diffie_hellman import DiffieHellman
from common.crypto.one_time_pad.component import OneTimePadContext
from common.crypto.one_time_pad.one_time_add import OneTimeAdd
from service.fed_config import FedConfig
from .aggregation_base import AggregationRootBase, AggregationLeafBase

# {
#     "method": "otp",
#     "key_bitlength": 128,
#     "data_type": "torch.Tensor",
#     "key_exchange": {
#         "key_bitlength": 3072,
#         "optimized": True
#     },
#     "csprng": {
#         "name": "hmac_drbg",
#         "method": "sha512",
#     }
# }


def split_bytes(x: bytes, out_shape: Tuple[int]):
    if len(out_shape) == 0:
        return int.from_bytes(x, 'big')
    elif len(out_shape) == 1:
        a = len(x) // out_shape[0]
        return [int.from_bytes(x[a * i: a * (i + 1)], 'big') for i in range(out_shape[0])]
    else:
        a = len(x) // out_shape[0]
        return [split_bytes(x[a * i: a * (i + 1)], out_shape[1:]) for i in range(out_shape[0])]


class AggregationOTPLeaf(AggregationLeafBase):
    def __init__(self, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> None:
        # super().__init__(sec_conf, root_id, leaf_ids)
        super().__init__(sec_conf, root_id, FedConfig.node_id)
        
        self.leaf_ids = leaf_ids or FedConfig.get_label_trainer() + FedConfig.get_trainer()
        leaf_pairs = combinations(self.leaf_ids, 2)
        # key exchange
        key_exchange_conf = sec_conf["key_exchange"]

        df_protocols: Dict[str, DiffieHellman] = {}

        for _leaf_ids in leaf_pairs:
            if Commu.node_id in _leaf_ids:
                df_protocol = DiffieHellman(list(_leaf_ids),
                                            key_bitlength=key_exchange_conf['key_bitlength'],
                                            optimized=key_exchange_conf["optimized"],
                                            channel_name="otp_diffie_hellman")
                df_protocols[df_protocol.chan.remote_id] = df_protocol

        entropys: Dict[str, bytes] = {remote_id: None for remote_id in df_protocols}

        # sequential
        # for id in df_protocols:
        #     entropys[id] = df_protocols[id].exchange(out_bytes=True)

        def func(id):
            entropys[id] = df_protocols[id].exchange(out_bytes=True)

        thread_list = []
        for id in df_protocols:
            task = threading.Thread(target=func, args=(id,))
            thread_list.append(task)

        for task in thread_list:
            task.start()

        for task in thread_list:
            task.join()

        # csprng
        csprng_conf = sec_conf["csprng"]

        self.csprngs: OrderedDict[str, DRBGBase] = OrderedDict()
        self.is_addition = []
        
        for remote_id in self.leaf_ids:
            if remote_id != Commu.node_id:
                self.csprngs[remote_id] = get_drbg_inst(name=csprng_conf["name"],
                                                        entropy=entropys[remote_id],
                                                        method=csprng_conf["method"],
                                                        nonce=b'',
                                                        additional_data=b'')
                self.is_addition.append(Commu.node_id < remote_id)

        # one-time-pad
        self.otp_context = OneTimePadContext(modulus_exp=sec_conf["key_bitlength"],
                                             data_type=sec_conf["data_type"])
                                             
    #@func_timer
    def _calc_upload_value(self, parameters: OrderedDict, parameters_weight: float) -> Tuple[OrderedDict, float]:
        # calculate total number of bytes of weights
        def f(t):
            return reduce(lambda x, y: x * y, t.shape, 1) * self.otp_context.modulus_exp // 8

        num_bytes_array = list(map(f, parameters.values()))
        csprng_generators = []
        for remote_id in self.csprngs:
            generator = self.csprngs[remote_id].generator(num_bytes=num_bytes_array,
                                                          additional_data=b'')
            csprng_generators.append(generator)

        weighted_parameters = OrderedDict()
        encrypted_parameters = OrderedDict()

        for k, v in parameters.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            weighted_parameters[k] = v * float(parameters_weight)
            # one_time_key = [np.array(split_bytes(bytes(next(g)), v.shape)) for g in csprng_generators]

            one_time_key = []
            for g in csprng_generators:
                x = bytearray(next(g))
                y = split_bytes(x, v.shape)
                one_time_key.append(np.array(y))

            encrypted_parameters[k] = OneTimeAdd.encrypt(context_=self.otp_context,
                                                         data=weighted_parameters[k],
                                                         one_time_key=one_time_key,
                                                         is_addition=self.is_addition,
                                                         serialized=False)
        return (encrypted_parameters, parameters_weight)


class AggregationOTPRoot(AggregationRootBase):
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
            received_value[0][0][k] = received_value[0][0][k].decode()
            received_value[0][0][k] /= total_weight
        return received_value[0][0]
