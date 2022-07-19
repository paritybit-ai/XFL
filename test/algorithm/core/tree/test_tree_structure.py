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


import logging

import pytest

from algorithm.core.tree.tree_structure import SplitInfo, Tree


def test_tree():
    tree = Tree(party_id='node-1')
    assert tree.party_id == 'node-1'
    assert len(list(tree.nodes.keys())) == 1
    assert list(tree.nodes.keys())[0] == tree.root_node_id
    assert tree.root_node.id == tree.root_node_id
    
    split_info = SplitInfo(
        owner_id='node-2',
        feature_idx=3,
        split_point=3.5,
        gain=5
    )
    
    node_id_1, node_id_2 = tree.split(node_id=tree.root_node_id,
                                      split_info=split_info,
                                      left_sample_index=[1, 3],
                                      right_sample_index=[0, 2],
                                      left_sample_weight=1.0,
                                      right_sample_weight=1.5)

    assert tree.check_node(node_id_1)
    assert tree.check_node(node_id_2)
    
    node_1 = tree.nodes[node_id_1]
    node_2 = tree.nodes[node_id_2]
    
    assert node_1.parent_node_id == tree.root_node_id
    assert node_2.parent_node_id == tree.root_node_id
    assert tree.root_node.left_node_id == node_id_1
    assert tree.root_node.right_node_id == node_id_2
    assert node_1.linkage == "left"
    assert node_2.linkage == "right"

    node_list = tree.search_nodes(depth=1)
    assert len(node_list) == 2
    assert node_1 in node_list
    assert node_2 in node_list
    
    node_id_1, node_id_2 = tree.split(node_id="11111",
                                      split_info=split_info,
                                      left_sample_index=[1, 3],
                                      right_sample_index=[0, 2],
                                      left_sample_weight=1.0,
                                      right_sample_weight=1.5)
    assert node_id_1 is None and node_id_2 is None

    
    
    
    
    