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


class TransferLogisticRegressionLabelTrainer(TransferLogisticRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, label=True, *args, **kwargs)
        self.set_seed(self.random_seed)
        self._init_model()
        self.metrics = self._set_metrics()
        self.export_conf = [{
            "class_name": "TransferLogisticRegression",
            "identity": self.identity,
            "filename": self.save_model_name,
            "num_classes": 2,
        }]

    def cal_phi_and_ua(self, dataloader):
        phi = None  # [1, hidden_features]  Φ_A
        ua = []
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            ua_batch = self.model(x_batch)  # [batch_size, hidden_features]
            ua.append(ua_batch)
            phi_batch = torch.sum(y_batch * ua_batch, axis=0).unsqueeze(0)
            if phi is None:
                phi = phi_batch
            else:
                phi += phi_batch
        
        ua = torch.concat(ua, axis=0)
        return phi, ua

    def cal_parameters(self, dual_channel):
        overlap_phi, overlap_ua = self.cal_phi_and_ua(self.overlap_train_dataloader)
        non_overlap_phi, non_overlap_ua = self.cal_phi_and_ua(self.non_overlap_train_dataloader)
        phi = (overlap_phi + non_overlap_phi) / self.sample_num

        phi_2 = torch.matmul(phi.T, phi)  # (Φ_A)‘(Φ_A) [hidden_features, hidden_features]

        overlap_y = self.overlap_y # {C(y)=y} [overlap_size, 1]
        overlap_y_2 = overlap_y * overlap_y  # {D(y)=y^2} [overlap_size, 1]

        # calculate 3 components will be sent to the trainer
        overlap_y_2_phi_2 = 0.25 * overlap_y_2.unsqueeze(2) * phi_2 # [overlap_size, hidden_features, hidden_features]
        overlap_y_phi = -0.5 * overlap_y * phi # [overlap_size, hidden_features]
        comp_ua = -self.constant_k * overlap_ua # [overlap_size, 1]

        # exchange components
        dual_channel.send((overlap_y_2_phi_2, overlap_y_phi, comp_ua))
        ub, ub_2, comp_ub = dual_channel.recv()

        # compute gradients to excute backward
        overlap_y_2_phi = (overlap_y_2 * phi).unsqueeze(1)
        loss_grads_const_part1 = 0.25 * torch.matmul(overlap_y_2_phi, ub_2).squeeze(1)
        loss_grads_const_part2 = overlap_y * ub
        const = torch.sum(loss_grads_const_part1, axis=0) - 0.5 * torch.sum(loss_grads_const_part2, axis=0)
    
        non_overlap_y = self.non_overlap_y
        non_overlap_ua_grad = self.alpha * const * non_overlap_y / self.sample_num
        overlap_ua_grad = self.alpha * const * overlap_y / self.sample_num + comp_ub

        # compute loss
        overlap_num = overlap_y.shape[0]
        overlap_loss = torch.sum(comp_ua * ub)

        ub_phi = torch.matmul(ub, phi.T)
        part1 = -0.5 * torch.sum(overlap_y * ub_phi)
        part2 = 1.0 / 8 * torch.sum(ub_phi * ub_phi)
        part3 = len(overlap_y) * np.log(2)
        loss_y = part1 + part2 + part3
        loss = self.alpha * (loss_y / overlap_num) + overlap_loss / overlap_num

        ua = torch.concat((overlap_ua, non_overlap_ua), axis=0)
        ua_grad = torch.concat((overlap_ua_grad, non_overlap_ua_grad), axis=0)

        # update phi
        self.phi = phi
        return loss, ua, ua_grad
        
    def train_loop(self, optimizer, lr_scheduler, dual_channel):
        loss_sum = 0
        for lepoch in range(1, self.local_epoch + 1):
            loss, ua, ua_grad = self.cal_parameters(dual_channel)
            optimizer.zero_grad()
            ua.backward(ua_grad)
            optimizer.step()
            loss_sum += loss
        if lr_scheduler:
            lr_scheduler.step()
            
        loss_sum /= self.local_epoch
        logger.info(f"loss: {loss_sum}")

    def val_loop(self, dual_channel):
        logger.info("val_loop start")
        self.model.eval()
        labels = []
        metric_output = {}
        for batch_idx, [y_batch] in enumerate(self.val_dataloader):
            labels.append(y_batch.numpy())
        labels = np.concatenate(labels, axis=0)
        ub = dual_channel.recv()
        predict_score = torch.matmul(ub, self.phi.T)
        predicts = torch.sigmoid(predict_score)
        val_predicts = np.array(predicts > 0.5, dtype=np.int32)

        metrics_conf: dict = self.train_params["metric"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](labels, val_predicts, **metrics_conf[method])
        logger.info(f"Metrics on validation set: {metric_output}")

    def fit(self):
        logger.info("Transfer logistic regression training start")
        dual_channel = DualChannel(
            name="transfer_logistic_regression_channel",
            ids=FedConfig.get_trainer()+[FedConfig.node_id]
        )

        optimizer_conf = self.train_params.get("optimizer", {})
        for k, v in optimizer_conf.items():
            optimizer = get_optimizer(k)(self.model.parameters(), **v
        )

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
        state_dict["phi"] = self.phi

        ModelPreserver.save(
            save_dir=self.save_dir, model_name=self.save_model_name,
            state_dict=state_dict, final=True
        )
