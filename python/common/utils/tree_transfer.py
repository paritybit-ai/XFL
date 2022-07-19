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


from typing import Dict

from algorithm.core.tree.tree_structure import Node, Tree
from common.utils.tree_pickle_structure import NodePickle, TreePickle


def label_trainer_tree_transfer(tree: Tree) -> TreePickle:
    """Transfer label trainer tree structure to pickle layout.

    Args:
        tree: Tree.

    Returns: TreePickle

    """
    nodes_pickle = {}
    for k, node in tree.nodes.items():
        nodes_pickle[k] = NodePickle(id=node.id, depth=node.depth, parent_node_id=node.parent_node_id,
                                     left_node_id=node.left_node_id, right_node_id=node.right_node_id,
                                     is_leaf=node.is_leaf,
                                     weight=node.weight, linkage=node.linkage, split_point=node.split_info.split_point,
                                     feature_idx=node.split_info.feature_idx,
                                     missing_value_on_left=node.split_info.missing_value_on_left,
                                     owner_id=node.split_info.owner_id) \
            if node.split_info else NodePickle(id=node.id, depth=node.depth, parent_node_id=node.parent_node_id,
                                               left_node_id=node.left_node_id, right_node_id=node.right_node_id,
                                               is_leaf=node.is_leaf, weight=node.weight, linkage=node.linkage,
                                               split_point=None, feature_idx=None, missing_value_on_left=None,
                                               owner_id=None)
    return TreePickle(party_id=tree.party_id, nodes=nodes_pickle, root_node_id=tree.root_node_id,
                      root_node=nodes_pickle[tree.root_node_id])


def trainer_tree_transfer(nodes: Dict[str, Node]) -> Dict[str, NodePickle]:
    """ Transfer trainer nodes structure to pickle layout.

    Args:
        nodes: Node.

    Returns: NodePickle.

    """
    nodes_pickle = {}
    for k, node in nodes.items():
        nodes_pickle[k] = NodePickle(id=node.id, depth=node.depth, parent_node_id=node.parent_node_id,
                                     left_node_id=node.left_node_id, right_node_id=node.right_node_id,
                                     is_leaf=node.is_leaf,
                                     weight=node.weight, linkage=node.linkage, split_point=node.split_info.split_point,
                                     feature_idx=node.split_info.feature_idx,
                                     missing_value_on_left=node.split_info.missing_value_on_left,
                                     owner_id=node.split_info.owner_id)
    return nodes_pickle
