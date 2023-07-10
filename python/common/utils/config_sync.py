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


import copy
from typing import Dict

from common.checker.matcher import get_matched_config
from common.communication.gRPC.python.channel import DualChannel
from common.utils.utils import update_dict
from service.fed_node import FedNode
from service.fed_config import FedConfig


class ConfigSynchronizer:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        
        assist_trainer = FedConfig.get_assist_trainer()
        label_trainers = FedConfig.get_label_trainer()
        trainers = FedConfig.get_trainer()
        
        assist_trainer = [assist_trainer] if assist_trainer else []
        all_trainers = assist_trainer + label_trainers + trainers
        self.coordinator = all_trainers[0]
        self.is_coordinator = FedNode.node_id == self.coordinator
        
        if self.is_coordinator:
            self.sync_chann: Dict[str, DualChannel] = {}
            for party_id in [id for id in all_trainers if id != self.coordinator]:
                self.sync_chann[party_id] = DualChannel(
                    name="sync_" + party_id, ids=[self.coordinator, party_id])
        else:
            self.sync_chann: DualChannel = None
            self.sync_chann = DualChannel(
                name="sync_" + FedNode.node_id, ids=[self.coordinator, FedNode.node_id]
            )
    
    def sync(self, sync_rule: dict):
        ''' for example:
            sync_rule = {
                "train_info": All()
            }
        '''
        def count_key(conf):
            if isinstance(conf, dict):
                num = len(conf.keys())
                for k, v in conf.items():
                    num += count_key(v)
                return num
            else:
                return 0

        if self.is_coordinator:
            conf_to_update = get_matched_config(self.config, sync_rule)
            max_key_num = count_key(conf_to_update)
            
            for party_id in self.sync_chann:
                conf = self.sync_chann[party_id].recv()
                num = count_key(conf)
                if num >= max_key_num:
                    conf_to_update = conf
                    max_key_num = num

            for party_id in self.sync_chann:
                self.sync_chann[party_id].send(conf_to_update)
        else:
            config_to_sync = get_matched_config(self.config, sync_rule)
            self.sync_chann.send(config_to_sync)
            conf_to_update = self.sync_chann.recv()
        update_dict(self.config, conf_to_update)
        return self.config
            