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
import os
import inspect
from functools import partial
from typing import OrderedDict
import torch.nn as nn

from algorithm.core.horizontal.aggregation.api import get_aggregation_root_inst
from algorithm.core.horizontal.aggregation.api import get_aggregation_leaf_inst
from algorithm.core.loss.torch_loss import get_lossfunc
from algorithm.core.metrics import get_metric
from algorithm.core.optimizer.torch_optimizer import get_optimizer
from algorithm.core.lr_scheduler.torch_lr_scheduler import get_lr_scheduler
from common.utils.config_parser import CommonConfigParser
from common.utils.algo_utils import earlyStoppingH
from common.utils.logger import logger
from common.utils.model_io import ModelIO
from algorithm.core.horizontal.template.hooker import Hooker


class BaseTrainer(Hooker):
    def __init__(self, train_conf: dict):
        Hooker.__init__(self)
        self.common_config = CommonConfigParser(train_conf)
        self.device = self.common_config.device
        # if self.common_config.early_stopping:
        self.earlystopping = earlyStoppingH(
            key=self.common_config.early_stopping.get("key", "acc"), 
            patience=self.common_config.early_stopping.get("patience", -1), 
            delta=self.common_config.early_stopping.get("delta", 0)
        )

        self.declare_hooks([
            "before_global_epoch", "before_local_epoch", "before_train_loop",
            "after_train_loop", "after_local_epoch", "after_global_epoch"
        ])
        self._init_context()

        self.model = self._set_model()
        self.train_dataloader = self._set_train_dataloader()
        self.val_dataloader = self._set_val_dataloader()
        self.lossfunc = self._set_lossfunc()
        self.optimizer = self._set_optimizer()
        self.lr_scheduler = self._set_lr_scheduler(self.optimizer)
        self.metrics = self._set_metrics()
        self.aggregator = self._set_aggregator(self.common_config.identity)

    def _init_context(self):
        self.context['g_epoch'] = 0
        self.context['l_epoch'] = 0
        self.context["config"] = self.common_config.config
        self.context["global_epoch_num"] = self.common_config.train_params.get("global_epoch", 0)
        self.context["local_epoch_num"] = self.common_config.train_params.get("local_epoch", 0)
        self.context["early_stop_flag"] = False
        self.context["early_stop_epoch"] = 0

    def _set_aggregator(self, party_type: str):
        if party_type == "assist_trainer":
            aggregator = get_aggregation_root_inst(self.common_config.encryption)
        else:
            aggregator = get_aggregation_leaf_inst(self.common_config.encryption)
        return aggregator

    def _set_model(self) -> nn.Module:
        raise NotImplementedError("The _set_model method is not implemented.")

    def _set_train_dataloader(self):
        raise NotImplementedError(
            "The _set_train_dataloader method is not implemented.")

    def _set_val_dataloader(self):
        raise NotImplementedError(
            "The _set_val_dataloader method is not implemented.")

    def _save_model(self, final: bool, context: dict):
        if not os.path.exists(self.common_config.output_dir):
            os.makedirs(self.common_config.output_dir)
        
        if final:
            if context["early_stop_flag"] & (context["early_stop_epoch"] > 0):
                if self.common_config.output_model_name != "":
                    ModelIO.copy_best_model(
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_model_name,
                        epoch=context["early_stop_epoch"],
                    )

                if self.common_config.output_onnx_model_name != "":
                    ModelIO.copy_best_model(
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_onnx_model_name,
                        epoch=context["early_stop_epoch"],
                    )
            
            else:
                if self.common_config.output_model_name != "":
                    ModelIO.save_torch_model(
                        state_dict=self.model.state_dict(),
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_model_name,
                    )

                if self.common_config.output_onnx_model_name != "":
                    input_dim = self.common_config.model_conf.get("input_dim")
                    if input_dim is None:
                        raise ValueError("input_dim is None")
                    ModelIO.save_torch_onnx(
                        model=self.model,
                        input_dim=(input_dim,),
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_onnx_model_name,
                    )
        
        else:
            if self.common_config.save_frequency == -1:
                return
            if context["g_epoch"] % self.common_config.save_frequency == 0:
                if self.common_config.output_model_name != "":
                    ModelIO.save_torch_model(
                        state_dict=self.model.state_dict(),
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_model_name,
                        epoch=context["g_epoch"],
                    )

                if self.common_config.output_onnx_model_name != "":
                    input_dim = self.common_config.model_conf.get("input_dim")
                    if input_dim is None:
                        raise ValueError("input_dim is None")
                    ModelIO.save_torch_onnx(
                        model=self.model,
                        input_dim=(input_dim,),
                        save_dir=self.common_config.output_dir,
                        model_name=self.common_config.output_onnx_model_name,
                        epoch=context["g_epoch"],
                    )
        
    def _load_model(self, context: dict):
        if self.common_config.pretrain_model_path != "":
            path = os.path.join(
                self.common_config.pretrain_model_path, 
                self.common_config.pretrain_model_name
            )
            state_dict = ModelIO.load_torch_model(path, device=self.device)
            self.model.load_state_dict(state_dict)

    def _set_optimizer(self):
        """ Define self.optimizer """
        optimizer_conf = OrderedDict(self.common_config.optimizer)
        optimizer = OrderedDict()

        for k, v in optimizer_conf.items():
            params = list(inspect.signature(get_optimizer(k)).parameters.values())
            accepted_keys = [param.name for param in params]
            v = {k: v[k] for k in v if k in accepted_keys}
            optimizer[k] = get_optimizer(k)(self.model.parameters(), **v)

        return optimizer

    def _set_lossfunc(self):
        """ Define self.lossfunc """
        lossfunc_conf = OrderedDict(self.common_config.lossfunc)
        lossfunc = OrderedDict()

        for k, v in lossfunc_conf.items():
            params = list(inspect.signature(get_lossfunc(k)).parameters.values())
            accepted_keys = [param.name for param in params]
            v = {k: v[k] for k in v if k in accepted_keys}
            lossfunc[k] = get_lossfunc(k)(**v)

        return lossfunc

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler_conf = OrderedDict(self.common_config.lr_scheduler)
        lr_scheduler = OrderedDict()
        for (k, v), o in zip(lr_scheduler_conf.items(), optimizer.values()):
            params = list(inspect.signature(get_lr_scheduler(k)).parameters.values())
            accepted_keys = [param.name for param in params]
            v = {k: v[k] for k in v if k in accepted_keys}
            lr_scheduler[k] = get_lr_scheduler(k)(o, **v)

        return lr_scheduler

    def _set_metrics(self):
        """ Define metric """
        metrics = {}
        metrics_conf: dict = self.common_config.metric

        for k, v in metrics_conf.items():
            params = list(inspect.signature(get_metric(k)).parameters.values())
            accepted_keys = [param.name for param in params]
            v = {k: v[k] for k in v if k in accepted_keys}
            metric = get_metric(k)
            metrics[k] = partial(metric, **v)
        return metrics

    def _state_dict_to_device(
            self, params: OrderedDict, device: str, inline: bool = True) -> OrderedDict:
        if not inline:
            params = copy.deepcopy(params)

        for k, v in params.items():
            params[k] = v.to(device)
        return params

    def train_loop(self):
        raise NotImplementedError("The train_loop method is not implemented.")

    def fit(self):
        global_epoch_num = self.context["global_epoch_num"]
        local_epoch_num = self.context["local_epoch_num"]
        self.execute_hook_at("before_global_epoch")

        for g_epoch in range(1, global_epoch_num + 1):
            logger.info(f"global epoch {g_epoch}/{global_epoch_num} start...")
            self.context['g_epoch'] = g_epoch
            if self.execute_hook_at("before_local_epoch"):
                break

            for l_epoch in range(1, local_epoch_num + 1):
                logger.info(
                    f"local epoch {l_epoch}/{local_epoch_num} of global epoch {g_epoch} start...")
                self.context['l_epoch'] = l_epoch
                self.execute_hook_at("before_train_loop")

                self.train_loop()

                self.execute_hook_at("after_train_loop")

                logger.info(
                    f"local epoch {l_epoch}/{local_epoch_num} of global epoch {g_epoch} finished.")

            if self.execute_hook_at("after_local_epoch"):
                break
            logger.info(f"global epoch {g_epoch}/{global_epoch_num} finished.")

        self.execute_hook_at("after_global_epoch")
