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


from typing import Dict, Optional


class NodePickle:
    def __init__(self, id: str, depth: int, parent_node_id: Optional[str], left_node_id: Optional[str],
                 right_node_id: Optional[str], is_leaf: bool, weight: Optional[float], linkage: Optional[str],
                 split_point: Optional[float], feature_idx: Optional[int], missing_value_on_left: Optional[bool],
                 owner_id: Optional[str]):
        super(NodePickle, self).__init__()
        self.id = id
        self.depth = depth
        self.parent_node_id = parent_node_id
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        self.is_leaf = is_leaf
        self.weight = weight
        self.linkage = linkage
        self.split_point = split_point
        self.feature_idx = feature_idx
        self.missing_value_on_left = missing_value_on_left
        self.owner_id = owner_id


class TreePickle:
    def __init__(self, party_id: str, nodes: Dict[str, NodePickle], root_node: NodePickle, root_node_id: str):
        super(TreePickle, self).__init__()
        self.party_id = party_id
        self.nodes = nodes
        self.root_node = root_node
        self.root_node_id = root_node_id
