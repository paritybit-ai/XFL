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
import random
from functools import reduce
from pathlib import Path
from common.checker.x_types import All

import numpy as np
import pandas as pd
import tenseal as ts
import torch

from common.checker.matcher import get_matched_config
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.algo_utils import earlyStopping
from common.utils.logger import logger
from common.utils.model_preserver import ModelPreserver
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from service.fed_node import FedNode
from service.fed_control import ProgressCalculator
from .base import VerticalPoissonRegressionBase


class VerticalPoissonRegressionLabelTrainer(VerticalPoissonRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        """
        Vertical Poisson Regression
        Args:
            train_conf: training parameters
            *args:
            **kwargs:
        """
        self.sync_channel = BroadcastChannel(name="sync")
        self._sync_config(train_conf)
        super().__init__(train_conf, label=True, *args, **kwargs)
        if self.random_seed is None:
            self.random_seed = random.randint(-(1 << 32), 1 << 32)
            self.sync_channel.broadcast(self.random_seed)
        self.set_seed(self.random_seed)

        self.progress_calculator = ProgressCalculator(self.global_epoch, len(self.train_dataloader))
        self._init_model(bias=True)
        self.export_conf = [{
            "class_name": "VerticalPoissonRegression",
            "identity": self.identity,
            "filename": self.save_model_name,
            "input_dim": self.data_dim,
            "bias": True,
            "version": "1.4.0"
        }]

        self.es = earlyStopping(key=self.early_stopping_config["key"],
                                patience=self.early_stopping_config["patience"],
                                delta=self.early_stopping_config["delta"])
        self.best_model = None
        self.broadcast_channel = BroadcastChannel(name="Public keys", root_id=FedConfig.get_assist_trainer())
        self.dual_channels = {"intermediate_label_trainer": {}, "gradients_loss": None}
        self.trainers = FedConfig.get_trainer()
        for party_id in self.trainers:
            self.dual_channels["intermediate_label_trainer"][party_id] = DualChannel(
                name="intermediate_label_trainer_" + party_id, ids=[FedConfig.node_id, party_id])
        self.dual_channels["gradients_loss"] = DualChannel(name="gradients_loss_" + FedConfig.node_id,
                                                           ids=[FedConfig.get_assist_trainer()] + [FedConfig.node_id])
        self.train_result = None
        self.val_result = None
        self.dual_channels["gradients_loss"].send(len(self.train_dataloader))
        self.dual_channels["gradients_loss"].send(self.global_epoch)
        self.dual_channels["gradients_loss"].send(self.batch_size)
        self.encryption_method = list(self.encryption_config.keys())[0].lower()
        self.dual_channels["gradients_loss"].send(self.encryption_config)
        self.dual_channels["gradients_loss"].send(self.encryption_method)

    def _sync_config(self, config):
        sync_rule = {
            "train_info": All()
        }
        config_to_sync = get_matched_config(config, sync_rule)
        self.sync_channel.broadcast(config_to_sync)

    def predict(self, input_data):
        pred_prob_epoch, y_epoch, pred_tmp_epoch = [], [], []
        for batch_idx, (x_batch, y_batch, _) in enumerate(input_data):
            pred_trainer_list = []
            pre_tmp = self.model(x_batch)
            # receive intermediate results from trainers
            for party_id in FedConfig.get_trainer():
                pred_trainer_list.append(self.dual_channels["intermediate_label_trainer"][party_id].recv(
                    use_pickle=True))
            # calculate prediction of batch and tmp_pred of batch
            pred_tmp_total = pre_tmp.numpy().astype(np.float32).flatten() + reduce(
                lambda x, y: x + y, pred_trainer_list)
            pred_total = np.exp(pred_tmp_total)
            # calculate prediction of epoch
            pred_prob_epoch += pred_total.tolist()
            pred_tmp_epoch += pred_tmp_total.tolist()
            y_epoch += y_batch.numpy().astype(np.float32).flatten().tolist()
        return y_epoch, pred_prob_epoch, pred_tmp_epoch

    def fit(self):
        self.check_data()
        public_context = None
        num_cores = -1
        rng = secrets.SystemRandom()

        logger.info("Vertical poisson regression training start")
        # receive encryption key from assist trainer
        if self.encryption_method == "ckks":
            logger.info("Receive ckks public key.")
            public_context = self.broadcast_channel.recv(use_pickle=False)
            public_context = ts.context_from(public_context)
            logger.info("Public key received.")
        elif self.encryption_method == "paillier":
            logger.info("Receive paillier public key.")
            public_context = self.broadcast_channel.recv(use_pickle=False)
            public_context = Paillier.context_from(public_context)
            logger.info("Public key received.")
        elif self.encryption_method == "plain":
            pass
        else:
            raise ValueError(f"Encryption method {self.encryption_method} not supported! Valid methods are "
                             f"'paillier', 'ckks', 'plain'.")
        # train
        for epoch in range(1, self.global_epoch + 1):
            loss_epoch = 0

            for batch_idx, (x_batch, y_batch, _) in enumerate(self.train_dataloader):
                regular_loss_tmp = 0
                regular_gradient_tmp = 0
                enc_regular_gradient_tmp = 0
                trainer_exp = None
                # calculate regular results
                if self.optimizer_config['p'] == 1:
                    regular_loss_tmp = torch.abs(self.model.linear.weight).sum() * self.optimizer_config['alpha']
                    regular_gradient_tmp = self.optimizer_config['alpha'] * (torch.abs(self.model.linear.weight)
                                                                             / self.model.linear.weight)
                elif self.optimizer_config['p'] == 2:
                    regular_loss_tmp = (self.model.linear.weight ** 2).sum() * self.optimizer_config['alpha'] / 2
                    regular_gradient_tmp = self.optimizer_config['alpha'] * self.model.linear.weight
                elif self.optimizer_config['p'] == 0:
                    pass

                # receive intermediate exp results from trainers and compute total_exp
                logger.info("Calculate predicted exp result of all trainers.")
                pred_label_tmp = self.model(x_batch)
                pred_label_trainer = torch.exp(pred_label_tmp)
                if self.encryption_method == "ckks":
                    trainer_exp = self.dual_channels["intermediate_label_trainer"][self.trainers[-1]].recv(
                        use_pickle=False)
                    trainer_exp = ts.ckks_vector_from(public_context, trainer_exp)
                elif self.encryption_method == "paillier":
                    trainer_exp = self.dual_channels["intermediate_label_trainer"][self.trainers[-1]].recv(
                        use_pickle=False)
                    trainer_exp = Paillier.ciphertext_from(public_context, trainer_exp)
                elif self.encryption_method == "plain":
                    trainer_exp = self.dual_channels["intermediate_label_trainer"][self.trainers[-1]].recv()
                total_exp = trainer_exp * pred_label_trainer.numpy().astype(np.float32).flatten()

                # receive immediate results from trainers
                logger.info("Calculate predicted result of all trainers.")
                if self.encryption_method == "ckks":
                    total_sum = ts.ckks_vector(public_context, pred_label_tmp.numpy().astype(np.float32).flatten())
                else:
                    total_sum = pred_label_tmp.numpy().astype(np.float32).flatten()
                for party_id in self.trainers:
                    if self.encryption_method == "ckks":
                        total_sum = total_sum + ts.ckks_vector_from(public_context, self.dual_channels[
                            "intermediate_label_trainer"][party_id].recv(use_pickle=False))
                        trainer_regular_loss = ts.ckks_vector_from(public_context, self.dual_channels[
                            "intermediate_label_trainer"][party_id].recv(use_pickle=False))
                    elif self.encryption_method == "paillier":
                        total_sum = total_sum + Paillier.ciphertext_from(public_context, self.dual_channels[
                            "intermediate_label_trainer"][party_id].recv(use_pickle=False))
                        trainer_regular_loss = Paillier.ciphertext_from(public_context, self.dual_channels[
                            "intermediate_label_trainer"][party_id].recv(use_pickle=False))
                    elif self.encryption_method == "plain":
                        total_sum = total_sum + self.dual_channels["intermediate_label_trainer"][party_id].recv()
                        trainer_regular_loss = self.dual_channels["intermediate_label_trainer"][party_id].recv()

                # calculate total loss
                logger.info("Calculate total loss.")
                enc_loss = total_exp - total_sum * y_batch.numpy().astype(np.float32).flatten()
                if self.encryption_method == "ckks":
                    regular_loss_tmp = ts.ckks_vector(public_context,
                                                      regular_loss_tmp.numpy().astype(np.float32).flatten())
                else:
                    regular_loss_tmp = regular_loss_tmp.numpy().astype(np.float32).flatten()
                enc_loss_batch = enc_loss + regular_loss_tmp + trainer_regular_loss

                # send total loss to assist_trainer
                logger.info("Send encrypted total loss to assist_trainer.")
                if self.encryption_method == "ckks":
                    self.dual_channels["gradients_loss"].send(enc_loss_batch.serialize(), use_pickle=False)
                elif self.encryption_method == "paillier":
                    self.dual_channels["gradients_loss"].send(Paillier.serialize(enc_loss_batch), use_pickle=False)
                elif self.encryption_method == "plain":
                    self.dual_channels["gradients_loss"].send(enc_loss_batch)
                # receive decrypted loss from assist_trainer
                logger.info("Receive decrypted total loss from assist_trainer.")
                loss_batch = self.dual_channels["gradients_loss"].recv()
                loss_batch = loss_batch / x_batch.shape[0]
                logger.info("Loss of {} batch is {}".format(batch_idx, loss_batch))
                loss_epoch += loss_batch * x_batch.shape[0]

                # calculate intermediate result d
                logger.info("Calculate intermediate result d.")
                enc_y = None
                if self.encryption_method == "ckks":
                    enc_y = ts.ckks_vector(public_context, y_batch.numpy().astype(np.float32).flatten())
                elif self.encryption_method == "paillier":
                    enc_y = Paillier.encrypt(public_context, y_batch.numpy().astype(np.float32).flatten(),
                                             precision=self.encryption_config[self.encryption_method][
                                                 "precision"], obfuscation=True, num_cores=num_cores)
                elif self.encryption_method == "plain":
                    enc_y = y_batch.numpy().astype(np.float32).flatten()
                enc_d = total_exp - enc_y

                # send intermediate result d to trainers
                logger.info("Send intermediate result d to trainers.")
                for party_id in self.trainers:
                    if self.encryption_method == "ckks":
                        self.dual_channels["intermediate_label_trainer"][party_id].send(enc_d.serialize(),
                                                                                        use_pickle=False)
                    elif self.encryption_method == "paillier":
                        self.dual_channels["intermediate_label_trainer"][party_id].send(Paillier.serialize(enc_d),
                                                                                        use_pickle=False)
                    elif self.encryption_method == "plain":
                        self.dual_channels["intermediate_label_trainer"][party_id].send(enc_d)

                # calculate gradient for label_trainer
                logger.info("Calculate gradients for label_trainer.")
                if self.encryption_method == "ckks":
                    enc_regular_gradient_tmp = ts.ckks_vector(public_context,
                                                              regular_gradient_tmp.numpy().astype(np.float32).flatten())
                elif self.encryption_method == "paillier":
                    enc_regular_gradient_tmp = Paillier.encrypt(
                        public_context, regular_gradient_tmp.numpy().astype(np.float32).flatten(),
                        precision=self.encryption_config[self.encryption_method]["precision"],
                        obfuscation=True, num_cores=num_cores)
                elif self.encryption_method == "plain":
                    enc_regular_gradient_tmp = regular_gradient_tmp.numpy().astype(np.float32).flatten()

                if self.encryption_method == "ckks":
                    gradient_label_trainer_w = enc_d.matmul(x_batch.numpy()) + enc_regular_gradient_tmp
                else:
                    gradient_label_trainer_w = np.matmul(enc_d.reshape(1, len(enc_d)), x_batch.numpy()
                                                         ) + enc_regular_gradient_tmp
                gradient_label_trainer_b = enc_d

                if self.encryption_method == "ckks":
                    # add noise to encrypted gradients and send to assist_trainer
                    logger.info("Calculate noised gradients for label_trainer.")
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    noise_b = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[0])],
                                       dtype=np.float32)
                    noise_b /= 100000
                    noised_gradient_label_trainer_w = gradient_label_trainer_w + noise
                    noised_gradient_label_trainer_b = gradient_label_trainer_b + noise_b
                    logger.info("Send noised gradient to assist_trainer.")
                    self.dual_channels["gradients_loss"].send(noised_gradient_label_trainer_w.serialize(),
                                                              use_pickle=False)
                    self.dual_channels["gradients_loss"].send(noised_gradient_label_trainer_b.serialize(),
                                                              use_pickle=False)
                    # receive decrypted gradient from assist_trainer
                    logger.info("Receive decrypted gradient from assist_trainer.")
                    noised_decrypt_gradient = self.dual_channels["gradients_loss"].recv()
                    noised_decrypt_gradient_label_trainer_w = noised_decrypt_gradient["noised_gradient_label_trainer_w"]
                    noised_decrypt_gradient_label_trainer_b = noised_decrypt_gradient["noised_gradient_label_trainer_b"]
                    gradient_label_trainer_w = noised_decrypt_gradient_label_trainer_w - noise
                    gradient_label_trainer_b = noised_decrypt_gradient_label_trainer_b - np.sum(noise_b)
                elif self.encryption_method == "paillier":
                    # add noise to encrypted gradients and send to assist_trainer
                    logger.info("Calculate noised gradients for label_trainer.")
                    noise = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[1])],
                                     dtype=np.float32)
                    noise /= 100000
                    noise_b = np.array([rng.randint(1 << 24, 1 << 26) - (1 << 25) for _ in range(x_batch.shape[0])],
                                       dtype=np.float32)
                    noise_b /= 100000
                    noised_gradient_label_trainer_w = gradient_label_trainer_w + noise
                    noised_gradient_label_trainer_b = gradient_label_trainer_b + noise_b
                    logger.info("Send noised gradient to assist_trainer.")
                    self.dual_channels["gradients_loss"].send(Paillier.serialize(noised_gradient_label_trainer_w),
                                                              use_pickle=False)
                    self.dual_channels["gradients_loss"].send(Paillier.serialize(noised_gradient_label_trainer_b),
                                                              use_pickle=False)
                    # receive decrypted gradient from assist_trainer
                    logger.info("Receive decrypted gradient from assist_trainer.")
                    noised_decrypt_gradient = self.dual_channels["gradients_loss"].recv()
                    noised_decrypt_gradient_label_trainer_w = noised_decrypt_gradient["noised_gradient_label_trainer_w"]
                    noised_decrypt_gradient_label_trainer_b = noised_decrypt_gradient["noised_gradient_label_trainer_b"]
                    gradient_label_trainer_w = noised_decrypt_gradient_label_trainer_w - noise
                    gradient_label_trainer_b = noised_decrypt_gradient_label_trainer_b - np.sum(noise_b)
                elif self.encryption_method == "plain":
                    gradient_label_trainer_b = gradient_label_trainer_b.sum()

                # update w and b of label_trainer
                gradient_label_trainer_w = gradient_label_trainer_w / x_batch.shape[0]
                gradient_label_trainer_b = gradient_label_trainer_b / x_batch.shape[0]
                logger.info("Update weights of label trainer.")
                self.model.linear.weight -= (torch.FloatTensor(gradient_label_trainer_w) * self.optimizer_config["lr"])
                self.model.linear.bias -= (gradient_label_trainer_b * self.optimizer_config["lr"])

                # calculate and update the progress of the training
                self.progress_calculator.cal_custom_progress(epoch, batch_idx+1)

            loss_epoch = loss_epoch * (1 / len(self.train))
            logger.info("Loss of {} epoch is {}".format(epoch, loss_epoch))

            # predict train and val results for metrics
            logger.info("Predict train weights of label trainer.")
            self.train_result = self.predict(self.train_dataloader)
            loss_train_met = {"loss": loss_epoch}
            self._calc_metrics(np.array(self.train_result[1], dtype=float), np.array(self.train_result[0]),
                               epoch, stage="train", loss=loss_train_met)
            logger.info("Predict val weights of label trainer.")
            self.val_result = self.predict(self.val_dataloader)
            loss_val = np.mean(
                np.array(self.val_result[1]) - np.array(self.val_result[0]) * np.array(self.val_result[2]))
            loss_val_met = {"loss": loss_val}  # no regular
            val_metrics = self._calc_metrics(np.array(self.val_result[1], dtype=float), np.array(self.val_result[0]),
                                             epoch, stage="val", loss=loss_val_met)

            # early stopping
            val_metrics["loss"] = - val_metrics["loss"]
            if self.early_stopping_config["patience"] > 0:
                early_stop_flag, best_model_flag = self.es(val_metrics)
            else:
                early_stop_flag, best_model_flag = False, True

            # update best model
            if best_model_flag:
                self.best_model = copy.deepcopy(self.model)
            # send flags to trainers
            for party_id in FedConfig.get_trainer():
                self.dual_channels["intermediate_label_trainer"][party_id].send(
                    [early_stop_flag, best_model_flag, self.early_stopping_config["patience"]], use_pickle=True)
            # if need to save results by epoch
            if self.save_frequency > 0 and epoch % self.save_frequency == 0:
                ModelPreserver.save(save_dir=self.save_dir, model_name=self.save_model_name,
                                    state_dict=self.model.state_dict(), epoch=epoch)
            # if early stopping, break
            if early_stop_flag:
                # update the progress of 100 to show the training is finished
                ProgressCalculator.finish_progress()
                break

        # save model for infer
        save_model_config(stage_model_config=self.export_conf, save_path=Path(self.save_dir))
        # if not early stopping, save probabilities and model
        self._save_prob()
        ModelPreserver.save(save_dir=self.save_dir, model_name=self.save_model_name,
                            state_dict=self.best_model.state_dict(), final=True)
        # calculate feature importance
        self._save_feature_importance(self.dual_channels)

    def _save_prob(self):
        if self.interaction_params.get("write_training_prediction"):
            self._write_prediction(self.train_result[1], self.train_result[0], self.train_ids,
                                   stage="train", final=True)
        if self.interaction_params.get("write_validation_prediction"):
            self._write_prediction(self.val_result[1], self.val_result[0], self.val_ids,
                                   stage="val", final=True)

    def check_data(self):
        dim_channel = BroadcastChannel(name="check_data_com", ids=[FedConfig.node_id] + FedConfig.get_trainer())
        n = self.data_dim
        dims = dim_channel.collect()
        for dim in dims:
            n += dim
        if n <= 0:
            raise ValueError("Number of the feature is zero. Stop training.")

    def _save_feature_importance(self, channel):
        res = {"owner_id": [], "fid": [], "importance": []}
        other_weight_list = []
        for party_id in FedConfig.get_trainer():
            other_weight_list.append(channel["intermediate_label_trainer"][party_id].recv(use_pickle=True))
        for (owner_id, weights) in other_weight_list:
            for fid, weight in enumerate(weights):
                res["owner_id"].append(owner_id)
                res["fid"].append(fid)
                res["importance"].append(float(weight))
        for fid, weight in enumerate(self.best_model.state_dict()["linear.weight"][0]):
            res["owner_id"].append(FedNode.node_id)
            res["fid"].append(fid)
            res["importance"].append(float(weight))
        res = pd.DataFrame(res).sort_values(by="importance", key=lambda col: np.abs(col), ascending=False)
        res.to_csv(Path(self.save_dir, self.output["feature_importance"]["name"]), header=True, index=False,
                   float_format="%.6g")
