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
import random
import secrets
from pathlib import Path

import numpy as np
import tenseal as ts
import torch

from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.logger import logger
from service.fed_node import FedNode
from common.utils.model_preserver import ModelPreserver
from common.utils.utils import save_model_config
from .base import VerticalLogisticRegressionBase


class VerticalLogisticRegressionTrainer(VerticalLogisticRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, label=False, *args, **kwargs)
        self._init_model()
        self.export_conf = [{
            "class_name": "VerticalLogisticRegression",
            "identity": self.identity,
            "filename": self.save_model_name,
            "input_dim": self.data_dim,
            "bias": False
        }]
        self.set_seed(self.extra_config["shuffle_seed"])
        self.best_model = None

    # @staticmethod
    # def decrypt(values):
    #     return values

    def fit(self):
        """ train model
        Model parameters need to be updated before fitting.
        """
        patient = -1
        encryption_config = self.aggregation_config["encryption"]
        encryption_method = encryption_config["method"].lower()

        logger.info("Vertical logistic regression training start")

        broadcast_channel = BroadcastChannel(name="vertical_logistic_regression_channel")

        public_context = None

        if encryption_method == "ckks":
            logger.debug("Receive ckks public key.")
            public_context = broadcast_channel.recv(use_pickle=False)
            public_context = ts.context_from(public_context)
            logger.debug("Public key received.")
        elif encryption_method == "paillier":
            logger.debug("Receive paillier public key.")
            public_context = broadcast_channel.recv(use_pickle=False)
            public_context = Paillier.context_from(public_context)
            logger.debug("Public key received.")
        elif encryption_method == "plain":
            pass
        else:
            raise ValueError(
                f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'ckks', 'plain'.")

        rng = secrets.SystemRandom()

        for epoch in range(1, self.global_epoch + 1):
            for batch_idx, (x_batch) in enumerate(self.train_dataloader):
                x_batch = x_batch[0].to(self.device)

                # compute theta_trainer * x_trainer
                pred_trainer = self.model(x_batch)

                # send predict result to label trainer.
                logger.debug("Send predict result to label trainer.")
                broadcast_channel.send(pred_trainer)

                if encryption_method == "ckks":
                    pred_residual = broadcast_channel.recv(use_pickle=False)
                    pred_residual = ts.ckks_vector_from(public_context, pred_residual)
                elif encryption_method == "paillier":
                    pred_residual = broadcast_channel.recv(use_pickle=False)
                    pred_residual = Paillier.ciphertext_from(public_context, pred_residual)
                elif encryption_method == "plain":
                    pred_residual = broadcast_channel.recv()

                logger.debug("Received prediction residual from label trainer.")

                # Compute gradients for trainer.
                logger.debug("Calculate gradients for trainer.")

                if encryption_method == "ckks":
                    # Add noise
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    x_batch_numpy = x_batch.numpy()
                    # avoid bug in seal ckks when a column is all zero
                    sign = 1 if random.randint(0, 1) == 1 else -1
                    x_batch_numpy[np.where(np.sum(x_batch_numpy, axis=0) == 0)] = 1e-7 * sign
                    ciphertext = pred_residual.matmul(x_batch_numpy)
                    noised_gradient_trainer_linear = ciphertext + noise
                    # Send to label trainer
                    broadcast_channel.send(noised_gradient_trainer_linear.serialize(), use_pickle=False)
                    gradient_trainer_linear = broadcast_channel.recv()
                    gradient_trainer_linear = np.array(gradient_trainer_linear, dtype=np.float32)
                    gradient_trainer_linear -= noise
                    gradient_trainer_linear = - gradient_trainer_linear / x_batch.shape[0]
                    gradient_trainer_linear = torch.FloatTensor(gradient_trainer_linear).unsqueeze(-1)
                elif encryption_method == "paillier":
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    # Add noise
                    ciphertext = np.matmul(pred_residual, x_batch.numpy())
                    noised_gradient_trainer_linear = ciphertext + noise
                    # Send to label trainer
                    broadcast_channel.send(Paillier.serialize(noised_gradient_trainer_linear), use_pickle=False)
                    gradient_trainer_linear = broadcast_channel.recv()
                    gradient_trainer_linear = np.array(gradient_trainer_linear, dtype=np.float32)
                    gradient_trainer_linear -= noise
                    gradient_trainer_linear = - gradient_trainer_linear / x_batch.shape[0]
                    gradient_trainer_linear = torch.FloatTensor(gradient_trainer_linear).unsqueeze(-1)
                elif encryption_method == "plain":
                    gradient_trainer_linear = -torch.mm(pred_residual.t(), x_batch) / x_batch.shape[0]
                    gradient_trainer_linear = gradient_trainer_linear.t()

                # Regular section
                gradient_trainer_linear = gradient_trainer_linear.t()
                if self.optimizer_config['p'] == 1:
                    gradient_trainer_linear += (self.optimizer_config['alpha'] * (
                            torch.abs(self.model.linear.weight) / self.model.linear.weight)) / x_batch.shape[0]
                elif self.optimizer_config['p'] == 2:
                    gradient_trainer_linear += (2 * self.optimizer_config['alpha'] * self.model.linear.weight) / \
                                               x_batch.shape[0]
                elif self.optimizer_config['p'] == 0:
                    gradient_trainer_linear += 0
                else:
                    raise NotImplementedError("Regular P={} not implement.".format(self.optimizer_config['p']))
                gradient_trainer_linear = gradient_trainer_linear.t()

                self.model.linear.weight -= (gradient_trainer_linear * self.optimizer_config["lr"]).t()
                logger.debug("Weights update completed.")

            for batch_idx, (x_batch) in enumerate(self.val_dataloader):
                x_batch = x_batch[0].to(self.device)
                pred_trainer = self.model(x_batch)

                broadcast_channel.send(pred_trainer)

            early_stop_flag, save_flag, patient = broadcast_channel.recv()
            if save_flag:
                self.best_model = copy.deepcopy(self.model)

            if early_stop_flag:
                ModelPreserver.save(save_dir=self.save_dir, model_name=self.save_model_name,
                                    state_dict=self.best_model.state_dict(), final=True)
                if self.save_probabilities:
                    self._save_prob(best_model=self.best_model, channel=broadcast_channel)
                return None

            if self.save_frequency > 0 and epoch % self.save_frequency == 0:
                ModelPreserver.save(save_dir=self.save_dir,
                                    model_name=self.save_model_name,
                                    state_dict=self.model.state_dict(),
                                    epoch=epoch)

        if patient <= 0:
            self.best_model = copy.deepcopy(self.model)
        save_model_config(stage_model_config=self.export_conf, save_path=Path(self.save_dir))
        ModelPreserver.save(save_dir=self.save_dir, model_name=self.save_model_name,
                            state_dict=self.best_model.state_dict(), final=True)
        if self.save_probabilities:
            self._save_prob(best_model=self.best_model, channel=broadcast_channel)

        self._save_feature_importance(broadcast_channel)

    def _save_prob(self, best_model, channel):
        for batch_idx, (x_batch) in enumerate(self.train_dataloader):
            x_batch = x_batch[0].to(self.device)
            pred_trainer = best_model(x_batch)
            channel.send(pred_trainer)

        for batch_idx, (x_batch) in enumerate(self.val_dataloader):
            x_batch = x_batch[0].to(self.device)
            pred_trainer = best_model(x_batch)
            channel.send(pred_trainer)

    def _save_feature_importance(self, channel):
        channel.send((FedNode.node_id, self.best_model.state_dict()["linear.weight"][0]))
