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
from typing import Dict, OrderedDict, Tuple

import numpy as np
import pandas as pd

from common.communication.gRPC.python.commu import Commu
from common.crypto.csprng.drbg import get_drbg_inst
from common.crypto.csprng.drbg_base import DRBGBase
from common.crypto.key_agreement.diffie_hellman import DiffieHellman
from common.crypto.one_time_pad.component import OneTimePadCiphertext, OneTimePadContext
from common.crypto.one_time_pad.one_time_add import OneTimeAdd
from .table_agg_base import TableAggregatorAbstractAssistTrainer, TableAggregatorAbstractTrainer


def split_bytes(x: bytes, out_shape: Tuple[int]):
    if len(out_shape) == 0:
        return int.from_bytes(x, 'big')
    elif len(out_shape) == 1:
        a = len(x) // out_shape[0]
        return [int.from_bytes(x[a * i: a * (i + 1)], 'big') for i in range(out_shape[0])]
    else:
        a = len(x) // out_shape[0]
        return [split_bytes(x[a * i: a * (i + 1)], out_shape[1:]) for i in range(out_shape[0])]


class TableAggregatorOTPTrainer(TableAggregatorAbstractTrainer):
    def __init__(self, sec_conf: dict, trainer_ids: list, *args, **kwargs) -> None:
        super().__init__(sec_conf=sec_conf, *args, **kwargs)
        trainer_pairs = combinations(trainer_ids, 2)

        # key exchange
        key_exchange_conf = sec_conf["key_exchange"]

        df_protocols: Dict[str, DiffieHellman] = {}

        for _trainer_ids in trainer_pairs:
            if Commu.node_id in _trainer_ids:
                df_protocol = DiffieHellman(list(_trainer_ids),
                                            key_bitlength=key_exchange_conf['key_bitlength'],
                                            optimized=key_exchange_conf["optimized"],
                                            channel_name="otp_diffie_hellman")
                df_protocols[df_protocol.chan.remote_id] = df_protocol

        entropys: Dict[str, bytes] = {remote_id: None for remote_id in df_protocols}

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

        csprng_conf = sec_conf["csprng"]

        self.csprngs: OrderedDict[str, DRBGBase] = OrderedDict()
        self.is_addition = []

        for remote_id in trainer_ids:
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

    def send(self, table: pd.Series) -> None:
        """

        Args:
            table:

        Returns:

        """
        def f(t):
            return reduce(lambda x, y: x * y, t.shape, 1) * self.otp_context.modulus_exp // 8

        if table is None:
            self.broadcast_chan.send(value=None)
            return

        num_bytes_array = list(map(f, table))
        csprng_generators = []
        for remote_id in self.csprngs:
            generator = self.csprngs[remote_id].generator(num_bytes=num_bytes_array,
                                                          additional_data=b'')
            csprng_generators.append(generator)
        one_time_key = []
        for g in csprng_generators:
            x = bytearray(next(g))
            y = split_bytes(x, table.shape)
            one_time_key.append(np.array(y))
        encrypted_table = OneTimeAdd.encrypt(context_=self.otp_context,
                                             data=table,
                                             one_time_key=one_time_key,
                                             is_addition=self.is_addition,
                                             serialized=False).data
        self.broadcast_chan.send(value=encrypted_table)


class TableAggregatorOTPAssistTrainer(TableAggregatorAbstractAssistTrainer):
    def __init__(self, sec_conf: dict, *args, **kwargs) -> None:
        super().__init__(sec_conf=sec_conf, *args, **kwargs)
        self.otp_context = OneTimePadContext(modulus_exp=sec_conf["key_bitlength"],
                                             data_type=sec_conf["data_type"])

    def aggregate(self) -> pd.Series:
        message = self.broadcast_chan.collect()
        ciphertext = None
        for table in message:
            if table is None:
                continue
            if ciphertext is None:
                ciphertext = OneTimePadCiphertext(data=table, context_=self.otp_context)
            else:
                ciphertext += OneTimePadCiphertext(data=table, context_=self.otp_context)
        ret = ciphertext.decode()
        return ret
