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


from typing import List, Tuple, Dict

import numpy as np
from ray.actor import ActorHandle


class RayDatasetInfo:
    def __init__(self):
        self.feature_names: List[str] = None
        self.label_name: str = None
        self.cat_names: List[str] = None
        self.shape: List[Tuple(int, int)] = None
        self.label: np.ndarray = None
        self.indices: np.ndarray = None
        self.actor_to_block_map: Dict[ActorHandle, int] = {}
        self.block_to_actor_map: Dict[int, ActorHandle] = {}
        self.blocks_shape: Dict[int, Tuple(int, int)] = {}
        
        self.split_points: Dict[str, list] = {}
        self.big_feature_names: List[str] = None
        
    def to_dict(self):
        res = {
            "feature_names": self.feature_names,
            "label_name": self.label_name,
            "shape": self.shape,
            "actor_to_block_map": self.actor_to_block_map,
            "block_to_actor_map": self.block_to_actor_map,
            "blocks_shape": self.blocks_shape
        }
        return res
    
    def to_tidy_dict(self):
        res = {
            "shape": self.shape,
            "actor_to_block_map": self.actor_to_block_map,
            "blocks_shape": self.blocks_shape
        }
        return res
