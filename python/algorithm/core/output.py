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


import os

from typing import Optional


class TableSaver():
    def __init__(self, path: str, name: Optional[str] = None):
        if name is None:
            splitted_path = path.split("/")
            self.path = '/'.join(splitted_path[:-1])
            self.name = splitted_path[-1]
        else:
            self.path = path
            self.name = name
            
    def save(self,
             epoch: int,
             data: dict,
             prefix: Optional[str] = None,
             suffix: Optional[str] = None,
             append: bool = True):
        name = ['.'.join(self.name.split('.')[:-1])]
        f_ext = self.name.split('.')[-1]
        if prefix is not None:
            name = [prefix] + name
        if suffix is not None:
            name = name + [suffix]
        name = '.'.join(['_'.join(name), f_ext])
        
        output_path = os.path.join(self.path, name)
        mode = 'a' if append else 'w'
    
        if os.path.exists(output_path):
            with open(output_path, mode) as f:
                features = []
                for k in data:
                    features.append("%.6g" % data[k])
                f.write("%d,%s\n" % (epoch, ','.join(features)))
                f.close()
        else:
            with open(output_path, mode) as f:
                f.write("%s,%s\n" % ("epoch", ','.join([_ for _ in data])))
                features = []
                for k in data:
                    features.append("%.6g" % data[k])
                f.write("%d,%s\n" % (epoch, ','.join(features)))
                f.close()
