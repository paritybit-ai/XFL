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


import pandas as pd
import numpy as np


class CsvReader(object):
    def __init__(self,
                 path: str,
                 has_id: bool = True,
                 has_label: bool = True):
        index_col = 0 if has_id else False
        self.table: pd.DataFrame = pd.read_csv(path, index_col=index_col)
        self.ids = self.table.index.to_numpy()
        self.table.reset_index(drop=True, inplace=True)
        self.has_id = has_id
        self.has_label = has_label
        self.label_col = 0 if has_label else -1

    def features(self, type: str = "numpy.ndarray"):
        if type == "numpy.ndarray":
            return self.table.iloc[:, self.label_col + 1:].to_numpy().astype(np.float32)
        else:  # pandas.dataframe
            return self.table.iloc[:, self.label_col + 1:]

    def label(self, type: str = "numpy.ndarray"):
        if self.label_col == 0:
            if type == "numpy.ndarray":
                return self.table.iloc[:, 0].to_numpy().astype(np.float32)
            else:
                return self.table.iloc[:, 0]
        else:
            return None

    def col_names(self):
        return self.table.columns.tolist()

    def feature_names(self):
        index = 0
        if self.has_label:
            index += 1
        return self.table.columns.tolist()[index:]

    def label_name(self):
        if self.label_col == 0:
            return self.table.columns.tolist()[0]
        else:
            return None


class NpzReader(object):
    def __init__(self,
                 path: str):
        self.data = np.load(path, allow_pickle=True)

    def features(self):
        return self.data["data"].astype(float)

    def label(self):
        return self.data["labels"].astype(float)

    
class NdarrayIterator():
    def __init__(self, data: np.ndarray, batch_size: int):
        self.data = data
        self.bs = batch_size
        self.index = 0
        
    def __len__(self):
        return len(self.data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            data = self.data[self.index: self.index + self.bs]
            self.index += self.bs
            return data
        else:
            self.index = 0
            raise StopIteration
