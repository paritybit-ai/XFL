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


import random
import string
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.utils.logger import logger


class SplitInfo(object):
    def __init__(self, 
                 owner_id: str,
                 feature_idx: Optional[int] = None, 
                 # is_category: bool, 
                 missing_value_on_left: Optional[bool] = None,
                 split_point: Optional[float] = None,
                 # left_split_categories: Optional[List[str]] = None,
                 gain: int = 0):
        self.owner_id = owner_id
        self.feature_idx = feature_idx
        # self.is_category = is_category
        self.missing_value_on_left = missing_value_on_left
        self.split_point = split_point
        # self.left_split_categories = left_split_categories # when is category
        self.gain = gain


class Node(object):
    def __init__(self, 
                 id: str, 
                 depth: int = -1,
                 sample_index: Optional[np.ndarray] = None,
                 left_node_id: Optional[str] = None, 
                 right_node_id: Optional[str] = None, 
                 parent_node_id: Optional[str] = None, 
                 split_info: Optional[SplitInfo] = None,
                 is_leaf: bool = True, 
                 weight: Optional[float] = None,
                 linkage: Optional[str] = None):
        self.id = id
        self.depth = depth
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        self.parent_node_id = parent_node_id
        self.split_info = split_info
        self.is_leaf = is_leaf
        self.weight = weight
        self.linkage = linkage

        # for training
        self.sample_index = sample_index

    def infer_tree_transfer(self):
        self.sample_index = None
        self.split_info.gain = None
        
    # def update_as_leaf(self, weight: float):
    #     self.weight = weight
    #     self.is_leaf = True
        
    def update_as_non_leaf(self,
                           split_info: SplitInfo, 
                           left_node_id: str,
                           right_node_id: str):
        self.split_info = split_info
        self.is_leaf = False
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        
    def clear(self):
        del self.non_missing_sample_index
        del self.missing_sample_index
        self.non_missing_sample_index = None
        self.missing_sample_index = None
        
                            
class Tree(object):
    def __init__(self, 
                 party_id: str, 
                 root_node_id: Optional[str] = None):
        self.party_id = party_id
        self.nodes: Dict[str, Node] = {}
        if root_node_id is None:
            self.root_node = Node(id=self._generate_id(), depth=0)
            self.root_node_id = self.root_node.id
        else:
            self.root_node = Node(id=root_node_id, depth=0)
            self.root_node_id = root_node_id
        self.nodes[self.root_node.id] = self.root_node

    def clear_training_info(self):
        for k, node in self.nodes.items():
            node.sample_index = None

    def check_node(self, node_id):
        if node_id not in self.nodes:
            return False
        return True
        
    def search_nodes(self, depth: int) -> List[Node]:
        res = []
        for node_id, node in self.nodes.items():
            if node.depth == depth:
                res.append(node)
        return res
    
    # def get_brother_node(self, node: Node) -> Node:
    #     if self.root_node.id == node.id:
    #         return None
        
    #     parent_node = self.nodes[node.parent_node_id]
    #     if parent_node.left_node_id == node.id:
    #         return self.nodes[parent_node.right_node_id]
    #     else:
    #         return self.nodes[parent_node.left_node_id]
                
    def _generate_id(self) -> str:
        flag = True
        while flag:
            id = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            if id not in self.nodes:
                flag = False
        return id
    
    def split(self,
              node_id: str, 
              split_info: SplitInfo, 
              left_sample_index: List[int],
              right_sample_index: List[int],
              left_sample_weight: float,
              right_sample_weight: float,
              left_node_id: Optional[str] = None,
              right_node_id: Optional[str] = None) -> Tuple[str, str]:
        if not self.check_node(node_id):
            logger.warning(f"Node_id {node_id} not valid, can't split")
            return None, None
        
        node = self.nodes[node_id]
        node.split_info = split_info
        node.is_leaf = False
        
        left_child_node = Node(id=self._generate_id() if left_node_id is None else left_node_id, 
                               depth=node.depth+1, 
                               sample_index=left_sample_index, 
                               parent_node_id=node_id,
                               weight=left_sample_weight,
                               linkage="left")
        
        right_child_node = Node(id=self._generate_id() if right_node_id is None else right_node_id, 
                                depth=node.depth+1, 
                                sample_index=right_sample_index,
                                parent_node_id=node_id,
                                weight=right_sample_weight,
                                linkage="right")
        
        self.nodes[left_child_node.id] = left_child_node
        self.nodes[right_child_node.id] = right_child_node
        node.left_node_id = left_child_node.id
        node.right_node_id = right_child_node.id
        return left_child_node.id, right_child_node.id
    
    def set_weight(self, node_id: str, weight: float):
        if self.check_node(node_id):
            node = self.nodes[node_id]
            node.weight = weight
        else:
            logger.warning(f"Node_id {node_id} not valid, can't set weight.")


# class BoostingTree(object):
#     def __init__(self, federal_info):
#         self.federal_info = federal_info
#         self.party_id_map = {}
#         self.trees = []
#         self.version = '0.1'
        
#     def append(self, tree: Tree):
#         self.trees.append(tree)
        
#     def write(self): # ?
#         # sample_index_list 设为[]
#         pass
    
#     def read(self, path):
#         pass
        
