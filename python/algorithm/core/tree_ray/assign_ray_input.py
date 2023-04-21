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


from typing import Optional, Dict

import numpy as np
from ray.actor import ActorHandle

from .dataset_info import RayDatasetInfo


def gen_actor_indices(indices: np.ndarray,
                      dataset_info: RayDatasetInfo):
    if indices is None:
        # out_indices_dict = {k: None for k in dataset_info.actor_to_block_map}
        out_indices_dict = {}
        for actor, blocks in dataset_info.actor_to_block_map.items():
            if len(blocks) != 0:
                out_indices_dict[actor] = None
            
    else:
        rows = sum([shape[0] for shape in dataset_info.shape])
        bool_indices = np.zeros((rows,), dtype='bool')
        bool_indices[indices] = True
        
        indices_dict: Dict[int, np.ndarray] = {}
        out_indices_dict: Dict[ActorHandle, Dict[int, np.ndarray]] = {}
        start_row = 0

        for block_id, shape in dataset_info.blocks_shape.items():
            end_row = start_row + shape[0]
            idx = np.where(bool_indices[start_row: end_row])[0]
            if len(idx) == 0:
                # Avoid the block where no sample is selected
                pass
            else:
                indices_dict[block_id] = idx + start_row
            start_row = end_row
        
        for actor, block_id_list in dataset_info.actor_to_block_map.items():
            idx = {k: indices_dict[k] for k in block_id_list if k in indices_dict}
            if idx != {}:
                # Avoid the actor if no data is selected for this actor
                out_indices_dict[actor] = idx
    return out_indices_dict
    

def gen_actor_input(indices: Optional[np.ndarray],
                    grad: Optional[np.ndarray],
                    hess: Optional[np.ndarray],
                    grad_hess: Optional[np.ndarray],
                    dataset_info: RayDatasetInfo) -> list:
    """ Generate indices, grad, hess, grad_hess for each actor.
        Only one of the grad_hess and (grad, hess) is used, depending on whether grad_hess is None.
        The size of grad, hess or grad_hess is equal to the size of indices.

    Args:
        indices (Optional[np.ndarray]): row indices selected for a tree node, the values should in ascend order.
        grad (Optional[np.ndarray]): grad
        hess (Optional[np.ndarray]): hess
        grad_hess (Optional[np.ndarray]): packed grad hess (ciphertext)
        dataset_info (RayDatasetInfo): dataset info

    Returns:
        list: list of indices, grad, hess, grad_hess for each actor
    """
    block_interval_map = {}
    
    if indices is None:
        # out_indices_dict = {k: None for k in dataset_info.actor_to_block_map}
        out_indices_dict = {}
        for actor, blocks in dataset_info.actor_to_block_map.items():
            if len(blocks) != 0:
                out_indices_dict[actor] = {actor: None}
        
        start_idx = 0
        for block_id, shape in dataset_info.blocks_shape.items():
            end_idx = start_idx + shape[0]
            block_interval_map[block_id] = [start_idx, end_idx]
            start_idx = end_idx
    else:
        rows = sum([shape[0] for shape in dataset_info.shape])
        bool_indices = np.zeros((rows,), dtype='bool')
        bool_indices[indices] = True
        
        indices_dict: Dict[int, np.ndarray] = {}
        out_indices_dict: Dict[ActorHandle, Dict[int, np.ndarray]] = {}
        start_row = 0
        start_idx = 0
        for block_id, shape in dataset_info.blocks_shape.items():
            end_row = start_row + shape[0]
            idx = np.where(bool_indices[start_row: end_row])[0]
            if len(idx) == 0:
                # Avoid the block where no sample is selected
                pass
            else:
                indices_dict[block_id] = idx + start_row
                end_idx = start_idx + len(idx)
                block_interval_map[block_id] = [start_idx, end_idx]
                start_idx = end_idx
            start_row = end_row
        
        for actor, block_id_list in dataset_info.actor_to_block_map.items():
            idx = {k: indices_dict[k] for k in block_id_list if k in indices_dict}
            if idx != {}:
                # Avoid the actor if no data is selected for this actor
                out_indices_dict[actor] = idx

    if grad_hess is not None:
        grad_hess_dict: Dict[int, np.ndarray] = {}
        out_grad_hess_dict: Dict[ActorHandle, Dict[int, np.ndarray]] = {}
        out_grad_dict = None
        out_hess_dict = None
        
        for block_id, (start_idx, end_idx) in block_interval_map.items():
            grad_hess_dict[block_id] = grad_hess[start_idx: end_idx]
        
        for actor, block_id_list in dataset_info.actor_to_block_map.items():
            idx = {k: grad_hess_dict[k] for k in block_id_list if k in grad_hess_dict}
            if idx != {}:
                out_grad_hess_dict[actor] = idx
    else:
        grad_dict: Dict[int, np.ndarray] = {}
        hess_dict: Dict[int, np.ndarray] = {}
        out_grad_dict: Dict[ActorHandle, Dict[int, np.ndarray]] = {}
        out_hess_dict: Dict[ActorHandle, Dict[int, np.ndarray]] = {}
        out_grad_hess_dict = None
        
        for block_id, (start_idx, end_idx) in block_interval_map.items():
            grad_dict[block_id] = grad[start_idx: end_idx]
            hess_dict[block_id] = hess[start_idx: end_idx]
        
        for actor, block_id_list in dataset_info.actor_to_block_map.items():
            idx_grad, idx_hess = {}, {}
            for k in block_id_list:
                if k in grad_dict:
                    idx_grad[k] = grad_dict[k]
                    idx_hess[k] = hess_dict[k]

            if idx_grad != {}:
                out_grad_dict[actor] = idx_grad
                out_hess_dict[actor] = idx_hess
    
    return out_indices_dict, out_grad_dict, out_hess_dict, out_grad_hess_dict