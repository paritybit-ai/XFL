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


import numpy as np
import torch

from common.communication.gRPC.python.channel import DualChannel
from common.utils.logger import logger
from service.fed_config import FedConfig
from .base import TransferLogisticRegressionBase
from algorithm.core.optimizer.torch_optimizer import get_optimizer
from algorithm.core.lr_scheduler.torch_lr_scheduler import get_lr_scheduler
from common.utils.model_preserver import ModelPreserver


class TransferLogisticRegressionTrainer(TransferLogisticRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, label=False, *args, **kwargs)
        self.set_seed(self.random_seed)
        self._init_model()
        self.export_conf = [{
            "class_name": "TransferLogisticRegression",
            "identity": self.identity,
            "filename": self.save_model_name,
            "num_classes": 2,
        }]

    def cal_ub(self, dataloader):
        ub = []
        for batch_idx, [x_batch] in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            ub_batch = self.model(x_batch)  # [batch_size, hidden_features]
            ub.append(ub_batch)

        ub = torch.concat(ub, axis=0)
        return ub

    def cal_parameters(self, dual_channel):
        ub = self.cal_ub(self.overlap_train_dataloader) # [overlap_size, hidden_features]

        # calculate 3 components will be sent to the trainer
        ub_ex = ub.unsqueeze(1)
        ub_2 = torch.matmul(ub.unsqueeze(2), ub_ex) # [overlap_size, hidden_features, hidden_features]
        comp_ub = -self.constant_k * ub # [overlap_size, hidden_features]

        # exchange components
        overlap_y_2_phi_2, overlap_y_phi, comp_ua = dual_channel.recv()
        dual_channel.send((ub, ub_2, comp_ub))

        # compute gradients to excute backward
        ub_overlap_y_2_phi_2 = torch.matmul(ub_ex, overlap_y_2_phi_2)
        l1_grad_b = ub_overlap_y_2_phi_2.squeeze(1) + overlap_y_phi
        ub_grad = self.alpha * l1_grad_b + comp_ua

        return ub, ub_grad

    def fit(self):
        logger.info("Transfer logistic regression training start")
        dual_channel = DualChannel(
            name="transfer_logistic_regression_channel",
            ids=FedConfig.get_label_trainer()+[FedConfig.node_id]
            )

        optimizer_conf = self.train_params.get("optimizer", {})
        for k, v in optimizer_conf.items():
            optimizer = get_optimizer(k)(self.model.parameters(), **v)

        lr_scheduler = None
        lr_scheduler_conf = self.train_params.get("lr_scheduler", {})
        for k, v in lr_scheduler_conf.items():
            lr_scheduler = get_lr_scheduler(k)(optimizer, **v)

        for epoch in range(1, self.global_epoch + 1):
            self.model.train()
            logger.info(f"trainer's global epoch {epoch}/{self.global_epoch} start...")
            self.train_loop(optimizer, lr_scheduler, dual_channel)
            self.val_loop(dual_channel)
    
        state_dict = self.model.state_dict()

        ModelPreserver.save(
            save_dir=self.save_dir, model_name=self.save_model_name,
            state_dict=state_dict, final=True
        )

    def train_loop(self, optimizer, lr_scheduler, dual_channel):
        for lepoch in range(1, self.local_epoch + 1):
            ub, ub_grad = self.cal_parameters(dual_channel)
            optimizer.zero_grad()
            ub.backward(ub_grad)
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

    def val_loop(self, dual_channel):
        self.model.eval()
        ub = self.cal_ub(self.val_dataloader)
        dual_channel.send(ub)
