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

import pandas as pd

from common.communication.gRPC.python.channel import BroadcastChannel
from common.communication.gRPC.python.commu import Commu
from service.fed_config import FedConfig


class TableAggregatorAbstractTrainer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, sec_conf: dict = None, *args, **kwargs) -> None:
        self.sec_conf = sec_conf
        self.broadcast_chan = BroadcastChannel(name='table',
                                               ids=Commu.trainer_ids,
                                               root_id=FedConfig.get_assist_trainer(),
                                               auto_offset=True)

    def send(self, table: pd.Series) -> None:
        pass


class TableAggregatorAbstractAssistTrainer(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, sec_conf: dict = None, *args, **kwargs) -> None:
        self.sec_conf = sec_conf
        self.broadcast_chan = BroadcastChannel(name='table',
                                               ids=Commu.trainer_ids,
                                               root_id=FedConfig.get_assist_trainer(),
                                               auto_offset=True)

    def aggregate(self) -> pd.Series:
        pass
