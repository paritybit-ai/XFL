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
from torch import Tensor
from copy import deepcopy
from typing import OrderedDict, List, Optional
from torch.optim.optimizer import Optimizer, required
from ..base import BaseTrainer
from algorithm.core.horizontal.aggregation.aggregation_base import AggregationLeafBase


class SCAFFOLDLabelTrainer(BaseTrainer):
    def __init__(self, train_conf: dict):
        self.prev_gmodel_params = None
        self.gmodel_params = None
        self.gmodel_grad = []
        self.lmodel_grad = []
        super().__init__(train_conf)
        
        self.register_hook(place="before_local_epoch", rank=1, func=self._download_model, desc="download global model")
        self.register_hook(place="before_local_epoch", rank=2, func=self._update_gmodel_grad, desc="Update gmodel grad")
        self.register_hook(place="after_local_epoch", rank=0, func=self._update_lmodel_grad, desc="Update lmodel grad")
        self.register_hook(place="after_local_epoch", rank=1, func=self._upload_model, desc="upload local model")
        
    def _download_model(self, context: dict):
        aggregator: AggregationLeafBase = self.aggregator
        new_state_dict = aggregator.download()
        
        self._state_dict_to_device(new_state_dict, self.device, inline=True)
        self.model.load_state_dict(new_state_dict)
        
    def _upload_model(self, context: dict):
        aggregator: AggregationLeafBase = self.aggregator
        if self.device != "cpu":
            state_dict = self._state_dict_to_device(self.model.state_dict(), "cpu", inline=False)
        else:
            state_dict = self.model.state_dict()
        aggregation_config = self.train_params["aggregation_config"]
        weight = aggregation_config.get("weight") or len(self.train_dataloader)
        aggregator.upload(state_dict, weight)

    def _update_gmodel_grad(self, context):
        self.gmodel_grad.clear()
        if self.gmodel_params:
            self.prev_gmodel_params = deepcopy(self.gmodel_params)
        self.gmodel_params = [p.data.detach().clone() for p in self.model.parameters()]
        if self.prev_gmodel_params:
            for w, prev_w in zip(self.gmodel_params, self.prev_gmodel_params):
                self.gmodel_grad.append(w.sub(prev_w))
        return

    def _update_lmodel_grad(self, context):
        if len(self.lmodel_grad) == 0:
            for l_w, g_w in zip(self.model.parameters(), self.gmodel_params):
                self.lmodel_grad.append(l_w.sub(g_w))
        else:
            for i in range(len(self.lmodel_grad)):
                self.lmodel_grad[i] += -self.gmodel_grad[i] + [p.data.detach() for p in self.model.parameters()][i] - self.gmodel_params[i]
        return

    def _set_optimizer(self):
        """ Define self.optimizer """
        optimizer_conf = OrderedDict(
            self.train_params.get("optimizer_config", {}))
        optimizer = OrderedDict()

        for k, v in optimizer_conf.items():
            optimizer[k] = SCAFFOLDOptimizer(self.model.parameters(), self.gmodel_grad, self.lmodel_grad,
            self.train_params.get("local_epoch", 0)*len(self.train_dataloader), **v)

        return optimizer


class SCAFFOLDOptimizer(Optimizer):

    def __init__(self, params, gmodel_grad, lmodel_grad, iter_num, lr=required, weight_decay=0, maximize=False,
    momentum=0, dampening=0, nesterov=False, amsgrad=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(gmodel_grad=gmodel_grad, lmodel_grad=lmodel_grad, iter_num=iter_num, lr_history=[], lr_sum=1, lr=lr,
        weight_decay=weight_decay, maximize=maximize, momentum=momentum, dampening=dampening, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        loss = None

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgdfold(params_with_grad, d_p_list, momentum_buffer_list, gmodel_grad=group['gmodel_grad'], lmodel_grad=group['lmodel_grad'], lr_sum=group['lr_sum'],
            lr=group['lr'], weight_decay=group['weight_decay'], maximize=group['maximize'],
            momentum=group['momentum'], dampening=group['dampening'], nesterov=group['nesterov'])
            group['lr_history'].append(group['lr'])
            if len(group['lr_history']) == group['iter_num']:
                group['lr_sum'] = sum(group['lr_history'])
                group['lr_history'].clear()

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        return loss

def sgdfold(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]],
gmodel_grad: List[Tensor], lmodel_grad:List[Tensor], lr_sum: float, lr: float, weight_decay: float, maximize: bool,
momentum: float, dampening: float, nesterov: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        alpha = lr if maximize else -lr
        beta = lr_sum if maximize else -lr_sum
        if len(gmodel_grad) > 0:
            param.add_(d_p - (lmodel_grad[i] - gmodel_grad[i]) / beta, alpha=alpha)
        else:
            param.add_(d_p, alpha=alpha)