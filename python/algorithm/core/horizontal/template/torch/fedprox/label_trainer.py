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


import torch
from typing import OrderedDict
from algorithm.core.loss.torch_loss import get_lossfunc
from ..base import BaseTrainer
from algorithm.core.horizontal.aggregation.aggregation_base import AggregationLeafBase


class FedProxLabelTrainer:
    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer
        self.mu = self.trainer.common_config.aggregation.get("mu", 0)
        
    def register(self):
        self.trainer.register_hook(
            place="before_local_epoch", rank=-3,
            func=self._sync_early_stop_flag, desc="sync early stop flag"
        )
        self.trainer.register_hook(
            place="before_local_epoch", rank=-2, func=self._download_model, 
            desc="download global model"
        )
        self.trainer.register_hook(
            place="before_local_epoch", rank=-1, func=self._update_gmodel_params, 
            desc="Update gmodel param"
        )
        self.trainer.register_hook(
            place="after_local_epoch", rank=-1, func=self._upload_model, 
            desc="upload local model"
        )

    # if get True, means the training is finished
    def _sync_early_stop_flag(self, context: dict):
        aggregator: AggregationLeafBase = self.trainer.aggregator
        early_stop_flag = aggregator.download()
        assert isinstance(early_stop_flag, bool)
        return early_stop_flag
        
    def _download_model(self, context: dict):
        aggregator: AggregationLeafBase = self.trainer.aggregator
        new_state_dict = aggregator.download()
        self.trainer._state_dict_to_device(new_state_dict, self.trainer.device, inline=True)
        self.trainer.model.load_state_dict(new_state_dict)

    def _upload_model(self, context: dict):
        aggregator: AggregationLeafBase = self.trainer.aggregator
        if self.trainer.device != "cpu":
            state_dict = self.trainer._state_dict_to_device(
                self.trainer.model.state_dict(), "cpu", inline=False
            )
        else:
            state_dict = self.trainer.model.state_dict()
        
        weight = self.trainer.common_config.aggregation.get("weight") or \
              len(self.trainer.train_dataloader)
        aggregator.upload(state_dict, weight)

    def _update_gmodel_params(self, context):
        self.gmodel_params = \
            [param.data.detach().clone() for param in self.trainer.model.parameters()]
        return

    def _set_lossfunc(self):
        """ Define self.lossfunc """
        lossfunc_conf = OrderedDict(self.trainer.common_config.lossfunc)
        lossfunc = OrderedDict()
        for k, v in lossfunc_conf.items():
            lossfunc[k] = self._get_fedprox_loss(k, v)
        return lossfunc
    
    def _get_fedprox_loss(self, k, v):
        def fedprox_loss(pred, label):
            reg = 0.0
            for w_prev, w in zip(self.gmodel_params, self.trainer.model.parameters()):
                reg += torch.pow(torch.norm(w - w_prev, p='fro'), 2)
            loss = get_lossfunc(k)(**v)(pred, label) + self.mu * reg / 2
            return loss
        return fedprox_loss