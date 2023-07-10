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
import torch.nn as nn
from typing import OrderedDict
from functools import partial
from common.utils.config_parser import CommonConfigParser
from algorithm.core.loss.torch_loss import get_lossfunc
from algorithm.core.metrics import get_metric
from algorithm.core.optimizer.torch_optimizer import get_optimizer
from algorithm.core.lr_scheduler.torch_lr_scheduler import get_lr_scheduler


class BaseTrainer:
    def __init__(self, train_conf: dict):
        self.common_config = CommonConfigParser(train_conf)
        if self.common_config.random_seed is not None:
            self.set_seed(self.common_config.random_seed)
        self.device = self.common_config.device
        self._parse_config()
        self.model = self._set_model()
        self._set_train_dataloader()
        self._set_val_dataloader()
        self.optimizer = self._set_optimizer()
        self.lr_scheduler = self._set_lr_scheduler(self.optimizer)
        self.lossfunc = self._set_lossfunc()
        self.optimizer = self._set_optimizer()
        self.lr_scheduler = self._set_lr_scheduler(self.optimizer)
        self.metrics = self._set_metrics()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _parse_config(self) -> nn.Module:
        raise NotImplementedError("The _parse_config method is not implemented.")
    
    def _set_optimizer(self):
        """ Define self.optimizer """
        optimizer_conf = OrderedDict(self.common_config.optimizer)
        optimizer = OrderedDict()

        for k, v in optimizer_conf.items():
            optimizer[k] = get_optimizer(k)(self.model.parameters(), **v)

        return optimizer

    def _set_lossfunc(self):
        """ Define self.lossfunc """
        lossfunc_conf = OrderedDict(self.common_config.lossfunc)
        lossfunc = OrderedDict()

        for k, v in lossfunc_conf.items():
            lossfunc[k] = get_lossfunc(k)(**v)

        return lossfunc

    def _set_lr_scheduler(self, optimizer: OrderedDict):
        lr_scheduler_conf = OrderedDict(self.common_config.lr_scheduler)
        lr_scheduler = OrderedDict()
        for (k, v), o in zip(lr_scheduler_conf.items(), optimizer.values()):
            lr_scheduler[k] = get_lr_scheduler(k)(o, **v)

        return lr_scheduler

    def _set_metrics(self):
        """ Define metric """
        metrics = {}
        metrics_conf: dict = self.common_config.metric

        for k, v in metrics_conf.items():
            metric = get_metric(k)
            metrics[k] = partial(metric, **v)
        return metrics
    
    def _set_model(self) -> nn.Module:
        raise NotImplementedError("The _set_model method is not implemented.")

    def _set_train_dataloader(self):
        raise NotImplementedError(
            "The _set_train_dataloader method is not implemented.")

    def _set_val_dataloader(self):
        raise NotImplementedError(
            "The _set_val_dataloader method is not implemented.")
