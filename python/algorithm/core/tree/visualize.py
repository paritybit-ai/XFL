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


from graphviz import Digraph

from .tree_structure import Tree


# def show_tree(tree: Tree):
#     dot = Digraph(name="tree", comment="tree vis", format="png")
    
#     for node_id, node in tree.nodes.items():
#         if node.split_info is None:
#             split_info = "%.5f" % node.weight
#         else:
#             if node.split_info.split_point is not None:
#                 # split_info = "%d:%.5f" % (node.split_info.feature_idx, node.split_info.split_point)
#                 split_info = "%d:%.5f:%s" % (node.split_info.feature_idx, node.split_info.split_point, node.split_info.owner_id)
#             else:
#                 split_info = "%d:%s" % (node.split_info.feature_idx, node.split_info.owner_id)
#         dot.node(name=node_id,
#                  label=split_info)
#                  # label="" if node.weight is None else str(node.weight))
#                 # label="" if node.sample_index is None else str(len(node.sample_index))) # + "," +  "" if node.weight is None else str(node.weight))
#                 # label = "split_point:" + "" if node.split_info is None else str(node.split_info.split_point) + \
#                 #             "num_samples:" + "" if node.sample_index is None else str(len(node.sample_index)))
        
#     for node_id, node in tree.nodes.items():
#         if node.left_node_id is not None:
#             dot.edge(node_id, node.left_node_id)
#             dot.edge(node_id, node.right_node_id)
    
#     dot.view(filename="tree_structure", directory='.')
