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


import copy

import pandas as pd

from .table_agg_base import TableAggregatorAbstractAssistTrainer
from .table_agg_base import TableAggregatorAbstractTrainer


class TableAggregatorPlainTrainer(TableAggregatorAbstractTrainer):
    def __init__(self, sec_conf: dict, *args, **kwargs) -> None:
        super().__init__(sec_conf=sec_conf, *args, **kwargs)
        
    def send(self, table: pd.Series) -> None:
        """

        Args:
            table:

        Returns:

        """
        self.broadcast_chan.send(value=table)
    

class TableAggregatorPlainAssistTrainer(TableAggregatorAbstractAssistTrainer):
    def __init__(self, sec_conf: dict, *args, **kwargs) -> None:
        super().__init__(sec_conf=sec_conf, *args, **kwargs)
        
    def aggregate(self) -> pd.Series:
        message = self.broadcast_chan.collect()
        ret = None
        for table in message:
            if table is None:
                continue
            if ret is None:
                ret = copy.deepcopy(table)
            else:
                ret += table
        return ret
