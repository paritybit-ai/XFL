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


from common.utils.tree_pickle_structure import NodePickle, TreePickle


class TestNodePickle():
    def test__init__(self):
        node = NodePickle("test", 2, "parent", "left", "right", True, 0.1, "linkage", 2.0, 2, True, "owner")
        assert node.id == "test"
        assert node.depth == 2
        assert node.parent_node_id ==  "parent"
        assert node.left_node_id == "left"
        assert node.right_node_id == "right"
        assert node.is_leaf == True
        assert node.weight == 0.1
        assert node.linkage == "linkage"
        assert node.split_point == 2.0
        assert node.feature_idx == 2
        assert node.missing_value_on_left == True
        assert node.owner_id == "owner"


class TestTreePickle():
    def test__init__(self):
        node = NodePickle("test", 2, "parent", "left", "right", True, 0.1, "linkage", 2.0, 2, True, "owner")
        tree = TreePickle("party_id", {}, node, "id")
        assert tree.party_id == "party_id"
        assert tree.nodes == {}
        assert tree.root_node == node
        assert tree.root_node_id == "id"

