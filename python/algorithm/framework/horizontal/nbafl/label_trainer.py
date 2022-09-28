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


from algorithm.core.horizontal.template.torch.fedtype import _get_label_trainer
from common.utils.logger import logger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from common.communication.gRPC.python.channel import DualChannel
from service.fed_config import FedConfig
import os
from algorithm.core.data_io import CsvReader
from .common import Common


class HorizontalNbaflLabelTrainer(Common, _get_label_trainer()):
    def __init__(self, train_conf: dict):
        _get_label_trainer().__init__(self, train_conf)
        self.sample_size_channel = DualChannel(
            name="sample_size_" + FedConfig.node_id,
            ids=[FedConfig.get_assist_trainer(), FedConfig.node_id]
        )
        # initialize prev params
        self.prev_params = [
            param.data.detach().clone() for param in self.model.parameters()
        ]
        self.mu = self.train_params['mu']
        self.delta = self.train_params['delta']
        self.c = np.sqrt(2 * np.log(1.25 / self.delta))
        self.epsilon = self.train_params['epsilon']

        # Update sample size
        self.register_hook(
            place="before_global_epoch", rank=2,
            func=self._update_sample_size, desc="Update local sample size"
        )
        # Calculate update sigma
        self.register_hook(
            place="before_global_epoch", rank=3,
            func=self._calc_uplink_sigma, desc="Calculate uplink sigma"
        )

        # Update prev param
        self.register_hook(
            place="after_local_epoch", rank=-2,
            func=self._update_prev_param, desc="Update prev param"
        )

        # Clip norm
        self.register_hook(
            place="after_local_epoch", rank=-1,
            func=self._clip_params, desc="Clip param norms"
        )

        # Add noise
        self.register_hook(
            place="after_local_epoch", rank=0,
            func=self._add_noise, desc="Add uplink noise"
        )

    def _update_prev_param(self, context):
        self.prev_params = [
            param.data.detach().clone() for param in self.model.parameters()
        ]

    def _cal_regularization(self, p=2):
        reg = 0.0
        for w_prev, w in zip(self.prev_params, self.model.parameters()):
            reg += torch.pow(torch.norm(w - w_prev, p), p)
        return self.mu * reg / p

    def _clip_params(self, context):
        for param in self.model.parameters():
            norm_ratio = torch.maximum(
                torch.ones(param.shape),
                torch.abs(param.data) / self.train_params['C']
            )
            param.data = param.data / norm_ratio
        return

    def _calc_uplink_sigma(self, context):
        delta_S_u = 2 * self.train_params['C'] / \
            len(self.train_dataloader.dataset)
        sigma_u = self.c * delta_S_u / self.epsilon
        logger.info("Uplink sigma: {}".format(sigma_u))
        self.sigma_u = sigma_u
        return

    def train_loop(self):
        self.model.train()
        train_loss = 0

        loss_func = next(iter(self.loss_func.items()))[1]
        optimizer = next(iter(self.optimizer.items()))[1]

        for batch_idx, (feature, label) in enumerate(self.train_dataloader):
            pred = self.model(feature)
            loss = loss_func(pred, label)
            reg = self._cal_regularization()
            loss += reg
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_dataloader)

        # retain current params
        self.prev_params = [
            param.data.detach().clone() for param in self.model.parameters()
        ]
        return train_loss

    def _add_noise(self, context):
        for param in self.model.parameters():
            param.data += torch.distributions.Normal(
                loc=0, scale=self.sigma_u).sample(param.size()).to(self.device)
        return

    def _update_sample_size(self, context):
        logger.info("trainset length: {}".format(
            len(self.train_dataloader.dataset)))
        self.sample_size_channel.send(len(self.train_dataloader.dataset))
        return
