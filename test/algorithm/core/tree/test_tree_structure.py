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


from typing import Type
import pytest

from algorithm.core.tree.tree_structure import BoostingTree, Node, NodeDict, SplitInfo, Tree


def test_node():
    node = Node(id="1111")
    node.update_as_non_leaf(split_info=None, left_node_id="12", right_node_id="23")
    node.update_as_leaf(weight=1.2)


def test_tree():
    with pytest.raises(ValueError):   
        Tree("1111", root_node_id="12", nodes={}, tree_index=0)
        
    tree = Tree(party_id='node-1', tree_index=0)
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
    
    Tree("1111", root_node_id=tree.root_node_id, nodes=tree.nodes, tree_index=0)
    
    tree.clear_training_info()
    for _, node in tree.nodes.items():
        assert node.sample_index is None
        
    tree.set_weight(node_id=tree.root_node_id, weight=-1)
    
    with pytest.raises(KeyError):
        tree.set_weight(node_id="aaa", weight=2)


def test_boosting_tree():
    with pytest.raises(TypeError):
        BoostingTree(lr=3, max_depth=[1])
    
    with pytest.raises(TypeError):
        BoostingTree(lr=[3, 4], max_depth=3)
    
    with pytest.raises(TypeError):
        BoostingTree(lr=[0.3, 0.3], max_depth=[3, 3], trees=None)
    
    with pytest.raises(ValueError):
        BoostingTree(lr=[0.1, 0.1], max_depth=[3, 3], trees=[None])
        
    with pytest.raises(ValueError):
        BoostingTree(lr=[0.1], max_depth=[3, 3], trees=[None])
    
    trees = {
        "lr": [0.3, 0.3],
        "max_depth": [5, 5],
        "suggest_threshold": 0.5,
        "num_trees": 2,
        "loss_method": "BCEWithLogitsLoss",
        "version": 1.0,
        "trees": [
            {
                "party_id": "A",
                "tree_index": 0,
                "root_node_id": "11111",
                "nodes": {
                    "11111": {
                        "id": "11111",
                        "depth": 0,
                        "left_node_id": "22222",
                        "right_node_id": "33333",
                        "is_leaf": False,
                        "weight": None,
                        "linkage": None,
                        "split_info": {
                            "owner_id": "A",
                            "feature_idx": 5,
                            "is_category": False,
                            "split_point": 1.23,
                            "left_cat": [],
                        }
                    },
                    "22222": {
                        "id": "22222",
                        "depth": 1,
                        "left_node_id": "44444",
                        "right_node_id": "55555",
                        "is_leaf": False,
                        "weight": None,
                        "linkage": 'left',
                        "split_info": {
                            "owner_id": "A",
                            "feature_idx": 1,
                            "is_category": True,
                            "split_point": None,
                            "left_cat": [3, 5],
                        }
                    },
                    "33333": {
                        "id": "33333",
                        "depth": 1,
                        "left_node_id": "66666",
                        "right_node_id": "77777",
                        "is_leaf": False,
                        "weight": None,
                        "linkage": 'right',
                        "split_info": {
                            "owner_id": "B",
                            "feature_idx": None,
                            "is_category": None,
                            "split_point": None,
                            "left_cat": None,
                        }
                    },
                    "44444": {
                        "id": "44444",
                        "depth": 2,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": 0,
                        "linkage": 'left',
                        "split_info": None
                    },
                    "55555": {
                        "id": "55555",
                        "depth": 2,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": 3.8,
                        "linkage": 'right',
                        "split_info": None
                    },
                    "66666": {
                        "id": "66666",
                        "depth": 2,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": -1.2,
                        "linkage": 'left',
                        "split_info": None
                    },
                    "77777": {
                        "id": "77777",
                        "depth": 2,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": 1.4,
                        "linkage": 'right',
                        "split_info": None
                    },
                }
            },
            {
                "party_id": "A",
                "tree_index": 1,
                "root_node_id": "a1111",
                "nodes": {
                    "a1111": {
                        "id": "a1111",
                        "depth": 0,
                        "left_node_id": "a2222",
                        "right_node_id": "a3333",
                        "is_leaf": False,
                        "weight": None,
                        "linkage": None,
                        "split_info": {
                            "owner_id": "B",
                            "feature_idx": 7,
                            "is_category": True,
                            "split_point": None,
                            "left_cat": [2, 4, 8],
                        }
                    },
                    "a2222": {
                        "id": "a2222",
                        "depth": 1,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": 12.52,
                        "linkage": 'left',
                        "split_info": None
                    },
                    "a3333": {
                        "id": "a3333",
                        "depth": 1,
                        "left_node_id": None,
                        "right_node_id": None,
                        "is_leaf": True,
                        "weight": -12.3,
                        "linkage": 'right',
                        "split_info": None
                    }
                }
            },
        ]
    }
    
    boosting_tree = BoostingTree.from_dict(trees)
    tree_dict = boosting_tree.to_dict()
    assert tree_dict == trees
    
    boosting_tree[:1]
    boosting_tree[1:]
    
    with pytest.raises(TypeError):
        boosting_tree[1]
    
    assert len(boosting_tree) == 2
    
    boosting_tree.append(tree=Tree.from_dict(trees["trees"][1]), 
                         lr=0.1,
                         max_depth=3)
    assert len(boosting_tree) == 3
    assert boosting_tree.lr == [0.3, 0.3, 0.1]
    assert boosting_tree.max_depth == [5, 5, 3]
    
    res = boosting_tree.to_dict(suggest_threshold=0.6,
                                compute_group=True)
    assert res["suggest_threshold"] == 0.6
    
    input_node_dict = {
            "a1111": {
                "id": "a1111",
                "depth": 0,
                "left_node_id": "a2222",
                "right_node_id": "a3333",
                "is_leaf": False,
                "weight": None,
                "linkage": None,
                "split_info": {
                    "owner_id": "B",
                    "feature_idx": 7,
                    "is_category": True,
                    "split_point": None,
                    "left_cat": [2, 4, 8],
                }
            },
            "a2222": {
                "id": "a2222",
                "depth": 1,
                "left_node_id": None,
                "right_node_id": None,
                "is_leaf": True,
                "weight": 12.52,
                "linkage": 'left',
                "split_info": None
            },
            "a3333": {
                "id": "a3333",
                "depth": 1,
                "left_node_id": None,
                "right_node_id": None,
                "is_leaf": True,
                "weight": -12.3,
                "linkage": 'right',
                "split_info": None
            }
        }

    node_dict = NodeDict({k: Node.from_dict(v) for k, v in input_node_dict.items()})
    node_dict.update(
        {
            "a2224": Node.from_dict(
                        {
                            "id": "a2224",
                            "depth": 1,
                            "left_node_id": None,
                            "right_node_id": None,
                            "is_leaf": True,
                            "weight": 12.52,
                            "linkage": 'left',
                            "split_info": None
                        }
                    )
        }
    )
    assert len(node_dict) == 4
    out = node_dict.to_dict()
    node_dict.from_dict(out)

