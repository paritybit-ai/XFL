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
import secrets
from pathlib import Path

import numpy as np
import tenseal as ts
import torch

from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.logger import logger
from service.fed_config import FedConfig
from service.fed_node import FedNode
from common.utils.model_preserver import ModelPreserver
from common.utils.utils import save_model_config
from .base import VerticalLinearRegressionBase


class VerticalLinearRegressionTrainer(VerticalLinearRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """[summary]

        Args:
            train_conf (dict): [description]
        """
        super().__init__(train_conf, label=False, *args, **kwargs)
        self._init_model()
        self.export_conf = [{
            "class_name": "VerticalLinearRegression",
            "identity": self.identity,
            "filename": self.save_model_name,
            "input_dim": self.data_dim,
            "bias": False
        }]
        if self.random_seed:
            self.set_seed(self.random_seed)
        self.best_model = None
        self.broadcast_channel = BroadcastChannel(name="Public keys", root_id=FedConfig.get_assist_trainer())
        if len(FedConfig.get_trainer()) > 1:
            self.broadcast_trainer = BroadcastChannel(name="Trainer exchange", root_id=FedConfig.node_id,
                                                      ids=FedConfig.get_trainer())
        self.dual_channels = {
            "intermediate_label_trainer": DualChannel(name="intermediate_label_trainer_" + FedConfig.node_id,
                                                      ids=FedConfig.get_label_trainer() + [FedConfig.node_id]),
            "gradients_loss": DualChannel(name="gradients_loss_" + FedConfig.node_id,
                                          ids=[FedConfig.get_assist_trainer()] + [FedConfig.node_id])
        }

    def predict(self, input_data):
        for batch_idx, (x_batch, _) in enumerate(input_data):
            # calculate prediction of batch
            pred_trainer = self.model(x_batch)
            # send to label_trainer
            self.dual_channels["intermediate_label_trainer"].send(pred_trainer.numpy().astype(np.float32).flatten(),
                                                                  use_pickle=True)

    def fit(self):
        """ train model
        Model parameters need to be updated before fitting.
        """
        self.check_data()
        num_cores = -1
        encryption_config = self.encryption_config
        encryption_method = list(self.encryption_config.keys())[0].lower()

        logger.info("Vertical linear regression training start")
        # receive encryption key from assist trainer
        public_context = None

        if encryption_method == "ckks":
            logger.info("Receive ckks public key.")
            public_context = self.broadcast_channel.recv(use_pickle=False)
            public_context = ts.context_from(public_context)
            logger.info("Public key received.")
        elif encryption_method == "paillier":
            logger.info("Receive paillier public key.")
            public_context = self.broadcast_channel.recv(use_pickle=False)
            public_context = Paillier.context_from(public_context)
            logger.info("Public key received.")
        elif encryption_method == "plain":
            pass
        else:
            raise ValueError(
                f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', 'ckks', 'plain'.")

        rng = secrets.SystemRandom()
        # train
        for epoch in range(1, self.global_epoch + 1):
            for batch_idx, (x_batch, _) in enumerate(self.train_dataloader):
                regular_loss_tmp = 0
                regular_gradient_tmp = 0
                enc_regular_gradient_tmp = 0
                # calculate regular results
                if self.optimizer_config['p'] == 1:
                    regular_loss_tmp = torch.abs(self.model.linear.weight) * self.optimizer_config['alpha']
                    regular_gradient_tmp = self.optimizer_config['alpha'] * (torch.abs(self.model.linear.weight)
                                                                             / self.model.linear.weight)
                elif self.optimizer_config['p'] == 2:
                    regular_loss_tmp = (self.model.linear.weight ** 2).sum() * self.optimizer_config['alpha'] / 2
                    regular_gradient_tmp = self.optimizer_config['alpha'] * self.model.linear.weight
                elif self.optimizer_config['p'] == 0:
                    pass

                # compute theta_trainer * x_trainer and loss of x_trainer
                pred_trainer = self.model(x_batch)
                square_tmp = (pred_trainer ** 2).sum() / 2
                loss_trainer = square_tmp + regular_loss_tmp

                # send intermediate results to label trainer.
                logger.info("Send intermediate result to label trainer.")
                enc_pred_trainer = None
                if encryption_method == "ckks":
                    enc_pred_trainer = ts.ckks_vector(public_context, pred_trainer.numpy().astype(np.float32).flatten())
                    enc_loss_trainer = ts.ckks_vector(public_context, loss_trainer.numpy().astype(np.float32).flatten())
                    self.dual_channels["intermediate_label_trainer"].send(enc_pred_trainer.serialize(),
                                                                          use_pickle=False)
                    self.dual_channels["intermediate_label_trainer"].send(enc_loss_trainer.serialize(),
                                                                          use_pickle=False)
                elif encryption_method == "paillier":
                    enc_pred_trainer = Paillier.encrypt(public_context,
                                                        pred_trainer.numpy().astype(np.float32).flatten(),
                                                        precision=encryption_config[encryption_method]["precision"],
                                                        obfuscation=True,
                                                        num_cores=num_cores)
                    enc_loss_trainer = Paillier.encrypt(public_context,
                                                        loss_trainer.numpy().astype(np.float32).flatten(),
                                                        precision=encryption_config[encryption_method]["precision"],
                                                        obfuscation=True,
                                                        num_cores=num_cores)
                    self.dual_channels["intermediate_label_trainer"].send(Paillier.serialize(enc_pred_trainer),
                                                                          use_pickle=False)
                    self.dual_channels["intermediate_label_trainer"].send(Paillier.serialize(enc_loss_trainer),
                                                                          use_pickle=False)
                elif encryption_method == "plain":
                    enc_pred_trainer = pred_trainer.numpy().astype(np.float32).flatten()
                    enc_loss_trainer = loss_trainer.numpy().astype(np.float32).flatten()
                    self.dual_channels["intermediate_label_trainer"].send(enc_pred_trainer, use_pickle=True)
                    self.dual_channels["intermediate_label_trainer"].send(enc_loss_trainer, use_pickle=True)

                # exchange theta_trainer * x_trainer to calculate loss_between_trainer when encryption is paillier
                logger.info("Calculate trainer_sum to label trainer when encryption is paillier.")
                if encryption_method == "paillier" and len(FedConfig.get_trainer()) > 1:
                    trainer_sum = 0
                    logger.info("Send intermediate result to other trainers when encryption is paillier.")
                    self.broadcast_trainer.broadcast(Paillier.serialize(enc_pred_trainer), use_pickle=False)
                    logger.info("Receive intermediate result from other trainers when encryption is paillier.")
                    trainer_tmp = self.broadcast_trainer.collect(use_pickle=False)
                    for trainer_u in trainer_tmp:
                        trainer_u = Paillier.ciphertext_from(public_context, trainer_u)
                        trainer_sum += np.sum(trainer_u * pred_trainer.numpy().astype(np.float32).flatten())
                    logger.info("Send trainer_sum to label trainer when encryption is paillier.")
                    self.dual_channels["intermediate_label_trainer"].send(Paillier.serialize(trainer_sum),
                                                                          use_pickle=False)

                # receive intermediate result d from label_trainer
                logger.info("Receive intermediate result d from label_trainer.")
                if encryption_method == "ckks":
                    enc_d = self.dual_channels["intermediate_label_trainer"].recv(use_pickle=False)
                    enc_d = ts.ckks_vector_from(public_context, enc_d)
                elif encryption_method == "paillier":
                    enc_d = self.dual_channels["intermediate_label_trainer"].recv(use_pickle=False)
                    enc_d = Paillier.ciphertext_from(public_context, enc_d)
                elif encryption_method == "plain":
                    enc_d = self.dual_channels["intermediate_label_trainer"].recv()

                # calculate gradient for trainer and send to assist_trainer
                logger.info("Calculate gradients for trainer.")
                if encryption_method == "ckks":
                    enc_regular_gradient_tmp = ts.ckks_vector(public_context,
                                                              regular_gradient_tmp.numpy().astype(np.float32).flatten())
                elif encryption_method == "paillier":
                    enc_regular_gradient_tmp = Paillier.encrypt(
                        public_context, regular_gradient_tmp.numpy().astype(np.float32).flatten(),
                        precision=encryption_config[encryption_method]["precision"],
                        obfuscation=True, num_cores=num_cores)
                elif encryption_method == "plain":
                    enc_regular_gradient_tmp = regular_gradient_tmp.numpy().astype(np.float32).flatten()

                if encryption_method == "ckks":
                    gradient_trainer_w = enc_d.matmul(x_batch.numpy()) + enc_regular_gradient_tmp
                else:
                    gradient_trainer_w = np.matmul(enc_d.reshape(1, len(enc_d)), x_batch.numpy()
                                                   ) + enc_regular_gradient_tmp

                # add noise to encrypted gradients and send to assist_trainer
                if encryption_method == "ckks":
                    logger.info("Calculate noised gradient for trainer.")
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    noised_gradient_trainer_w = gradient_trainer_w + noise
                    logger.info("Send noised gradient to assist_trainer.")
                    self.dual_channels["gradients_loss"].send(noised_gradient_trainer_w.serialize(), use_pickle=False)
                    # receive decrypted gradient from assist_trainer
                    logger.info("Receive decrypted gradient from assist_trainer.")
                    noised_gradient_trainer_w = self.dual_channels["gradients_loss"].recv()
                    gradient_trainer_w = noised_gradient_trainer_w - noise
                elif encryption_method == "paillier":
                    logger.info("Calculate noised gradient for trainer.")
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    noised_gradient_trainer_w = gradient_trainer_w + noise
                    logger.info("Send noised gradient to assist_trainer.")
                    self.dual_channels["gradients_loss"].send(Paillier.serialize(noised_gradient_trainer_w),
                                                              use_pickle=False)
                    # receive decrypted gradient from assist_trainer
                    logger.info("Receive decrypted gradient from assist_trainer.")
                    noised_gradient_trainer_w = self.dual_channels["gradients_loss"].recv()
                    gradient_trainer_w = noised_gradient_trainer_w - noise
                # gradient_trainer_w = torch.FloatTensor(gradient_trainer_w).unsqueeze(-1)

                # update w and b of trainer
                gradient_trainer_w = gradient_trainer_w / x_batch.shape[0]
                logger.info("Update weights of trainer.")
                self.model.linear.weight -= (torch.FloatTensor(gradient_trainer_w) * self.optimizer_config["lr"])

            # predict train and val for metrics
            logger.info("Predict train weights of trainer.")
            self.predict(self.train_dataloader)
            logger.info("Predict val weights of trainer.")
            self.predict(self.val_dataloader)

            # receive flags
            early_stop_flag, best_model_flag, patient = self.dual_channels["intermediate_label_trainer"].recv(
                use_pickle=True)
            # update best model
            if best_model_flag:
                self.best_model = copy.deepcopy(self.model)
            # if need to save results by epoch
            if self.save_frequency > 0 and epoch % self.save_frequency == 0:
                ModelPreserver.save(save_dir=self.save_dir,
                                    model_name=self.save_model_name,
                                    state_dict=self.model.state_dict(),
                                    epoch=epoch)
            # if early stopping, break
            if early_stop_flag:
                break

        # save model for infer
        save_model_config(stage_model_config=self.export_conf, save_path=Path(self.save_dir))
        # if not early stopping, save model
        ModelPreserver.save(save_dir=self.save_dir, model_name=self.save_model_name,
                            state_dict=self.best_model.state_dict(), final=True)
        # send w to label trainer
        self._save_feature_importance(self.dual_channels["intermediate_label_trainer"])

    def _save_feature_importance(self, channel):
        channel.send((FedNode.node_id, self.best_model.state_dict()["linear.weight"][0]))

    def check_data(self):
        dim_channel = BroadcastChannel(name="check_data_com", ids=[FedConfig.node_id] + FedConfig.get_trainer())
        dim_channel.send(self.data_dim)
