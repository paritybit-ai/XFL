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
                 is_category: bool = False,
                 missing_value_on_left: Optional[bool] = True,
                 split_point: Optional[float] = None,
                 left_cat: Optional[List[str]] = None,
                 gain: int = 0):
        self.owner_id = owner_id
        self.feature_idx = feature_idx
        self.is_category = is_category
        self.missing_value_on_left = missing_value_on_left
        self.split_point = split_point
        self.left_cat = left_cat  # when is category
        self.gain = gain
        
    @classmethod
    def from_dict(self, data: dict):
        split_info = SplitInfo(owner_id=data['owner_id'])
        for k, v in data.items():
            setattr(split_info, k, v)
        return split_info
        
    def to_dict(self):
        res = {}
        attribute_list = ["owner_id", "feature_idx", "is_category", "split_point", "left_cat"]
        for name in attribute_list:
            res[name] = getattr(self, name)
        return res
    
    # def to_min_dict(self):
    #     res = {}
    #     attribute_list = ["feature_idx", "is_category", "split_point", "left_cat"]
    #     for name in attribute_list:
    #         res[name] = getattr(self, name)
    #     return res


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
        
    @classmethod
    def from_dict(self, data: dict):
        node = Node(id=data["id"])
        for k, v in data.items():
            if k == "split_info":
                split_info = SplitInfo.from_dict(v) if v else None
                setattr(node, k, split_info)
            else:
                setattr(node, k, v)
        return node
    
    def to_dict(self):
        res = {}
        attribute_list = ["id", "depth", "left_node_id", "right_node_id",
                          "split_info", "is_leaf", "weight", "linkage"]
        for name in attribute_list:
            if name == "split_info":
                res[name] = getattr(self, name)
                if res[name]:
                    res[name] = res[name].to_dict()
            else:
                res[name] = getattr(self, name)
        return res
    
    def to_min_dict(self):
        res = {}
        res["id"] = self.id
        res["split_info"] = self.split_info.to_dict() if self.split_info else None
        return res
        
    def update_as_non_leaf(self,
                           split_info: SplitInfo,
                           left_node_id: str,
                           right_node_id: str):
        self.split_info = split_info
        self.is_leaf = False
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        
    def update_as_leaf(self, weight: float):
        self.weight = weight
        self.is_leaf = True

                            
class Tree(object):
    def __init__(self,
                 party_id: str,
                 tree_index: int,
                 root_node_id: Optional[str] = None,
                 nodes: Optional[Dict[str, Node]] = {}):
        self.party_id = party_id
        self.tree_index = tree_index
        self.root_node_id = root_node_id
        self.nodes: Dict[str, Node] = {}  # important
        self.nodes.update(nodes)
        
        if root_node_id is None:
            self.root_node = Node(id=self._generate_id(), depth=0)
            self.root_node_id = self.root_node.id
            self.nodes[self.root_node.id] = self.root_node
        else:
            if self.root_node_id not in self.nodes:
                raise ValueError(f"Tree root node id {self.root_node_id} not in nodes ids.")
            else:
                self.root_node = self.nodes[self.root_node_id]
    
    @classmethod
    def from_dict(cls, data: dict):
        tree = Tree(party_id=data["party_id"], tree_index=data["tree_index"])
        for k, v in data.items():
            if k == "nodes":
                value = {node_id: Node.from_dict(node)for node_id, node in v.items()}
                setattr(tree, k, value)
            else:
                setattr(tree, k, v)
        return tree
    
    def to_dict(self):
        res = {}
        attribute_list = ["party_id", "tree_index", "root_node_id", "nodes"]
        for name in attribute_list:
            if name == "nodes":
                nodes_dict: dict[str, Node] = getattr(self, name)
                res[name] = {k: v.to_dict() for k, v in nodes_dict.items()}
            else:
                res[name] = getattr(self, name)
        return res

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
                
    def _generate_id(self) -> str:
        flag = True
        while flag:
            id = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            if id not in self.nodes:
                flag = False
        return '_'.join([str(self.tree_index), id])
    
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
            raise KeyError(f"Node_id {node_id} not valid, can't set weight.")
            

class BoostingTree(object):
    """ For trainer with label
    """
    def __init__(self,
                 lr: List[float] = [],
                 max_depth: List[int] = [],
                 trees: List[Tree] = [],
                 suggest_threshold: Optional[float] = None,
                 loss_method: str = "BCEWithLogitsLoss",
                 version: str = '1.0'):
        if not isinstance(lr, list):
            raise TypeError(f"Parameters lr should be a list of values, not {type(lr)}.")
        
        if not isinstance(max_depth, list):
            raise TypeError(f"Parameters max_depth should be a list of values, not {type(max_depth)}.")
        
        if not isinstance(trees, list):
            raise TypeError(f"Parameters trees should be a list of Tree instance, not {type(trees)}")
        
        if len(lr) != len(trees):
            raise ValueError(f"Length of lr {len(lr)} not equals to the length of trees {len(trees)}.")
        
        if len(max_depth) != len(trees):
            raise ValueError(f"Length of max_depth {len(max_depth)} not equals to the length of trees {len(trees)}.")

        if loss_method not in ["BCEWithLogitsLoss"]:
            raise NotImplementedError(f"Method {loss_method} is not implemented.")

        self.trees = trees
        self.lr = lr
        self.max_depth = max_depth
        self.suggest_threshold = suggest_threshold
        self.loss_method = loss_method
        self.version = version
        
    @classmethod
    def from_dict(cls, data: dict):
        tree = BoostingTree(lr=data["lr"],
                            max_depth=data.get("max_depth", None),
                            trees=[Tree.from_dict(tree) for tree in data["trees"]],
                            suggest_threshold=data.get("suggest_threshold", None),
                            loss_method=data.get("loss_method", None),
                            version=data.get('version', '1.0'))
        return tree
    
    def to_dict(self,
                suggest_threshold: Optional[float] = None,
                compute_group: bool = False):
        res = {}
        # for binary classification
        res["suggest_threshold"] = suggest_threshold or self.suggest_threshold

        attribute_list = ["lr", "max_depth", "trees", "version", "loss_method"]
        for name in attribute_list:
            if name == "trees":
                trees = [tree.to_dict() for tree in getattr(self, name)]
                res[name] = trees
            else:
                res[name] = getattr(self, name)
                
        res['num_trees'] = len(res['trees'])
                
        if compute_group:
            node_id_of_owner = {}
            for tree in self.trees:
                for node_id in tree.nodes:
                    split_info = tree.nodes[node_id].split_info
                    owner_id = split_info.owner_id if split_info else None
                    if owner_id is None:
                        continue
                    if owner_id not in node_id_of_owner:
                        node_id_of_owner[owner_id] = [node_id]
                    else:
                        node_id_of_owner[owner_id].append(node_id)

            for owner_id in node_id_of_owner:
                node_id_of_owner[owner_id].sort()
                
            # For reducing transmit data at inference stage
            node_id_group = {}
            for _, v in node_id_of_owner.items():
                # Because owner_id is unstable
                node_id_group[v[0]] = v
            
            res["node_id_group"] = node_id_group
        
        return res
        
    def append(self, tree: Tree, lr: float, max_depth: int):
        self.trees.append(tree)
        self.lr.append(lr)
        self.max_depth.append(max_depth)
        
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, index):
        # support slice
        cls = type(self)
        if isinstance(index, slice):
            trees = self.trees[index]
            lr = self.lr[index]
            max_depth = self.max_depth[index]
            # Note suggest_threshold is probably not correct after slice
            suggest_threshold = self.suggest_threshold
            loss_method = self.loss_method
            version = self.version
            return cls(lr, max_depth, trees, suggest_threshold, loss_method, version)
        else:
            msg = "{cls.__name__} indices must be slice."
            raise TypeError(msg.format(cls=cls))
        
        
class NodeDict(object):
    """ For trainer without label, only store nodes
    """
    def __init__(self, nodes: Dict[str, Node] = None):
        self.nodes = nodes or {}
        
    @classmethod
    def from_dict(self, data: dict):
        nodes = {id: Node.from_dict(node) for id, node in data.items()}
        return NodeDict(nodes)
    
    def to_dict(self) -> dict:
        res = {id: node.to_min_dict() for id, node in self.nodes.items()}
        return res
        
    def update(self, nodes: Dict[str, Node]):
        for id, node in nodes.items():
            self.nodes[id] = node
    
    def __len__(self):
        return len(self.nodes)

