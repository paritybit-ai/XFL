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
from functools import partial
from typing import OrderedDict


from algorithm.core.horizontal.aggregation.api import get_aggregation_root_inst
from algorithm.core.horizontal.aggregation.api import get_aggregation_leaf_inst
from algorithm.core.loss import get_lossfunc
from algorithm.core.metrics import get_metric
from algorithm.core.optimizer import get_optimizer
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from ..hooker import Hooker
import tensorflow.keras as keras

class BaseTrainer(Hooker, TrainConfigParser):
    def __init__(self, train_conf: dict):
        Hooker.__init__(self)
        TrainConfigParser.__init__(self, train_conf)

        self.declare_hooks(["before_global_epoch", "before_local_epoch", "before_train_loop",
                            "after_train_loop", "after_local_epoch", "after_global_epoch"])

        self.model = self._set_model()
        self.train_dataloader = self._set_train_dataloader()
        self.val_dataloader = self._set_val_dataloader()
        self.loss_func = self._set_lossfunc()
        self.optimizer = self._set_optimizer()
        self.metrics = self._set_metrics()
        self.aggregator = self._set_aggregator(self.identity)

    def _set_aggregator(self, party_type: str):
        aggregation_config = self.train_params.get("aggregation_config", {})
        encryption_params = aggregation_config.get("encryption")

        #logger.info(encryption_params)

        if party_type == "assist_trainer":
            aggregator = get_aggregation_root_inst(encryption_params)
        else:
            aggregator = get_aggregation_leaf_inst(encryption_params)
        return aggregator

    def _set_model(self) -> keras.Model:
        raise NotImplementedError("The _set_model method is not implemented.")

    def _set_train_dataloader(self):
        raise NotImplementedError(
            "The _set_train_dataloader method is not implemented.")

    def _set_val_dataloader(self):
        raise NotImplementedError(
            "The _set_val_dataloader method is not implemented.")

    def _save_model(self, context: dict):
        path = self.output["model"]["path"]
        name = self.output["model"]["name"]
        type = self.output["model"]["type"]
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, name)
        if type == "file":
            self.model.save_weights(path)
        else:
            raise NotImplementedError(f"Type {type} not supported.")

    def _load_model(self, context: dict):
        pretrain_model_conf = self.input["pretrain_model"]
        if pretrain_model_conf != {}:
            path = os.path.join(
                pretrain_model_conf["path"], pretrain_model_conf["name"])
            self.model.load_weights(path)
            

    def _set_optimizer(self):
        """ Define self.optimizer """
        optimizer_conf = OrderedDict(
            self.train_params.get("optimizer_config", {}))
        optimizer = OrderedDict()

        for k, v in optimizer_conf.items():
            optimizer[k] = get_optimizer(k, framework="tf")(**v)

        return optimizer

    def _set_lossfunc(self):
        """ Define self.loss_func """
        loss_func_conf = OrderedDict(
            self.train_params.get("lossfunc_config", {}))
        loss_func = OrderedDict()

        for k, v in loss_func_conf.items():
            loss_func[k] = get_lossfunc(k, framework="tf")(**v)

        return loss_func


    def _set_metrics(self):
        """ Define metric """
        metrics = {}
        metrics_conf: dict = self.train_params.get("metric_config", {})

        for k, v in metrics_conf.items():
            metric = get_metric(k)
            metrics[k] = partial(metric, **v)
        return metrics


    def train_loop(self):
        raise NotImplementedError("The train_loop method is not implemented.")

    def fit(self):
        current_epoch = 1
        self.context["current_epoch"] = current_epoch
        self.context["train_conf"] = self.train_conf
        global_epoch_num = self.train_params.get("global_epoch", 0)
        local_epoch_num = self.train_params.get("local_epoch", 0)
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

                current_epoch += 1
                self.context["current_epoch"] = current_epoch
                logger.info(
                    f"local epoch {l_epoch}/{local_epoch_num} of global epoch {g_epoch} finished.")

            if self.execute_hook_at("after_local_epoch"):
                break
            logger.info(f"global epoch {g_epoch}/{global_epoch_num} finished.")

        self.execute_hook_at("after_global_epoch")
