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
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tenseal as ts
import torch
from sklearn.metrics import confusion_matrix
import random
import pickle

from common.checker.matcher import get_matched_config
from common.checker.x_types import All
from common.communication.gRPC.python.channel import BroadcastChannel
from common.crypto.paillier.paillier import Paillier
from common.evaluation.metrics import ThresholdCutter
from common.utils.algo_utils import earlyStopping
from common.utils.logger import logger
from common.utils.model_io import ModelIO
from common.utils.utils import save_model_config
from service.fed_node import FedNode
from service.fed_control import ProgressCalculator
from .base import VerticalLogisticRegressionBase
from .base import BLOCKCHAIN


class VerticalLogisticRegressionLabelTrainer(VerticalLogisticRegressionBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        self.sync_channel = BroadcastChannel(name="sync")
        self._sync_config(train_conf)
        super().__init__(train_conf, label=True, *args, **kwargs)
        if self.random_seed is None:
            self.random_seed = random.randint(-(1 << 32), 1 << 32)
            self.sync_channel.broadcast(self.random_seed)
            if BLOCKCHAIN:
                logger.debug(
                    f"Broadcast random seed, SHA256: {hashlib.sha256(pickle.dumps(self.random_seed)).hexdigest()}")
        self.set_seed(self.random_seed)

        self.progress_calculator = ProgressCalculator(self.global_epoch, len(self.train_dataloader))
        self._init_model(bias=True)
        self.export_conf = [{
            "class_name": "VerticalLogisticRegression",
            "identity": self.identity,
            "filename": self.save_onnx_model_name,
            "input_dim": self.data_dim,
            "bias": True,
            "version": "1.4.0",
            "input_schema": self.schema,
        }]

        self.es = earlyStopping(key=self.early_stopping_config["key"],
                                patience=self.early_stopping_config["patience"],
                                delta=self.early_stopping_config["delta"])
        self.best_model = None
        self.best_prediction_val = None
        self.best_prediction_train = None

    def _sync_config(self, config):
        sync_rule = {
            "train_info": {
                "interaction_params": All(),
                "train_params": {
                    "global_epoch": All(),
                    "batch_size": All(),
                    "encryption": All(),
                    "optimizer": All(),
                    "early_stopping": All(),
                    "random_seed": All()
                }
            }
        }

        config_to_sync = get_matched_config(config, sync_rule)
        self.sync_channel.broadcast(config_to_sync)
        if BLOCKCHAIN:
            logger.debug(
                f"Sync config, SHA256: {hashlib.sha256(pickle.dumps(config_to_sync)).hexdigest()}")

    def fit(self):
        self.check_data()
        logger.debug("Vertical logistic regression training start")
        broadcast_channel = BroadcastChannel(
            name="vertical_logistic_regression_channel")

        encryption_config = self.encryption_config
        # encryption_method = encryption_config["method"].lower()
        encryption_method = list(self.encryption_config.keys())[0].lower()

        private_context = None
        num_cores = -1
        pred_prob_list, y_list = [], []
        if encryption_method == "ckks":
            private_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=encryption_config[encryption_method]["poly_modulus_degree"],
                coeff_mod_bit_sizes=encryption_config[encryption_method]["coeff_mod_bit_sizes"]
            )
            private_context.generate_galois_keys()
            private_context.generate_relin_keys()
            private_context.global_scale = 1 << encryption_config[
                encryption_method]["global_scale_bit_size"]

            serialized_public_context = private_context.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True
            )

            logger.debug("Broadcast ckks public keys.")
            broadcast_channel.broadcast(
                serialized_public_context, use_pickle=False)
            if BLOCKCHAIN:
                logger.debug(
                    f"SHA256: {hashlib.sha256(serialized_public_context).hexdigest()}")
            logger.debug("Broadcast completed.")
        elif encryption_method == "paillier":
            num_cores = - \
                1 if encryption_config[encryption_method]["parallelize_on"] else 1
            private_context = Paillier.context(encryption_config[encryption_method]["key_bit_size"],
                                               djn_on=encryption_config[encryption_method]["djn_on"])
            logger.debug("Broadcast paillier public keys.")
            serialized_public_context = private_context.to_public().serialize()
            broadcast_channel.broadcast(
                serialized_public_context, use_pickle=False)
            if BLOCKCHAIN:
                logger.debug(
                    f"SHA256: {hashlib.sha256(serialized_public_context).hexdigest()}")
            logger.debug("Broadcast completed.")
        elif encryption_method == "plain":
            pass
        else:
            raise ValueError(f"Encryption method {encryption_method} not supported! Valid methods are 'paillier', "
                             f"'ckks', 'plain'.")

        loss_func = torch.nn.BCELoss()

        for epoch in range(1, self.global_epoch + 1):
            training_cm = np.zeros((2, 2))
            training_pred_prob_list, training_y_list, training_metric = [], [], {}
            for batch_idx, (x_batch, y_batch, _) in enumerate(self.train_dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # compute theta_scheduler * x_scheduler
                pred_label_trainer = self.model(x_batch)

                # collect predict result from trainers.
                pred_trainer_list = broadcast_channel.collect()
                logger.debug("Received predictions from trainers, length of collect list is {}."
                             .format(len(pred_trainer_list)))
                if BLOCKCHAIN:
                    logger.debug(
                        f"SHA256: {hashlib.sha256(pickle.dumps(pred_trainer_list)).hexdigest()}")

                # Add predictions.
                pred_total = torch.clone(pred_label_trainer)
                for pred_trainer in pred_trainer_list:
                    pred_total += pred_trainer
                pred_total = torch.sigmoid(pred_total)

                logger.debug("Aggregated predictions.")
                # Calculate gradients.
                pred_residual = y_batch - pred_total

                if encryption_method == "ckks":
                    enc_pred_residual = ts.ckks_vector(
                        private_context, pred_residual.numpy().flatten())
                    serialized_enc_pred_residual = enc_pred_residual.serialize()
                    broadcast_channel.broadcast(
                        serialized_enc_pred_residual, use_pickle=False)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Broadcast encrypted pred residual, SHA256: {hashlib.sha256(serialized_enc_pred_residual).hexdigest()}")
                elif encryption_method == "paillier":
                    enc_pred_residual = Paillier.encrypt(private_context,
                                                         pred_residual.numpy().astype(np.float32).flatten(),
                                                         precision=encryption_config[encryption_method]["precision"],
                                                         obfuscation=True,
                                                         num_cores=num_cores)
                    serialized_enc_pred_residual = Paillier.serialize(
                        enc_pred_residual)
                    broadcast_channel.broadcast(
                        serialized_enc_pred_residual, use_pickle=False)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Broadcast encrypted pred residual, SHA256: {hashlib.sha256(serialized_enc_pred_residual).hexdigest()}")
                elif encryption_method == "plain":
                    broadcast_channel.broadcast(pred_residual)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Broadcast pred residual, SHA256: {hashlib.sha256(pickle.dumps(pred_residual)).hexdigest()}")

                training_pred_prob_list += torch.squeeze(
                    pred_total, dim=-1).tolist()
                training_y_list += torch.squeeze(y_batch, dim=-1).tolist()
                if self.echo_training_metrics:
                    pred_total = (pred_total > 0.5).float()
                    training_cm += confusion_matrix(
                        y_true=y_batch.detach().numpy(), y_pred=pred_total.detach().numpy())

                # Gradients for label trainer.
                logger.debug("Calculate gradients for label trainer.")
                if self.optimizer_config['p'] == 1:
                    gradient_label_trainer_linear = -torch.mm(pred_residual.t(), x_batch) / x_batch.shape[0] + (
                        self.optimizer_config['alpha'] * (torch.abs(self.model.linear.weight)
                                                          / self.model.linear.weight)
                    ) / x_batch.shape[0]
                elif self.optimizer_config['p'] == 2:
                    gradient_label_trainer_linear = -torch.mm(pred_residual.t(), x_batch) / x_batch.shape[0] + (
                        2 * self.optimizer_config['alpha'] * self.model.linear.weight) / x_batch.shape[0]
                elif self.optimizer_config['p'] == 0:
                    gradient_label_trainer_linear = - \
                        torch.mm(pred_residual.t(), x_batch) / x_batch.shape[0]
                else:
                    raise NotImplementedError(
                        "Regular P={} not implement.".format(self.optimizer_config['p']))
                gradient_label_trainer_bias = -torch.mean(pred_residual, dim=0)
                gradient_label_trainer_linear = gradient_label_trainer_linear.t()

                # Collect trainers noise gradients, decrypt and broadcast.
                if encryption_method == "ckks":
                    gradient_list_trainer_linear = broadcast_channel.collect(
                        use_pickle=False)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Collect gradient list, SHA256: {hashlib.sha256(pickle.dumps(gradient_list_trainer_linear)).hexdigest()}")
                    gradient_list_trainer_linear = [ts.ckks_vector_from(private_context, i).decrypt() for i in
                                                    gradient_list_trainer_linear]
                    broadcast_channel.scatter(gradient_list_trainer_linear)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Scatter gradient, SHA256: {hashlib.sha256(pickle.dumps(gradient_list_trainer_linear)).hexdigest()}")
                elif encryption_method == "paillier":
                    gradient_list_trainer_linear = broadcast_channel.collect(
                        use_pickle=False)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Collect random seed, SHA256: {hashlib.sha256(pickle.dumps(gradient_list_trainer_linear)).hexdigest()}")
                    gradient_list_trainer_linear = [
                        Paillier.decrypt(private_context, Paillier.ciphertext_from(None, c), dtype='float',
                                         num_cores=num_cores) for c in gradient_list_trainer_linear]
                    broadcast_channel.scatter(gradient_list_trainer_linear)
                    if BLOCKCHAIN:
                        logger.debug(
                            f"Scatter gradient, SHA256: {hashlib.sha256(pickle.dumps(gradient_list_trainer_linear)).hexdigest()}")
                elif encryption_method == "plain":
                    pass

                self.model.linear.weight -= (gradient_label_trainer_linear *
                                             self.optimizer_config["lr"]).t()
                self.model.linear.bias -= (gradient_label_trainer_bias *
                                           self.optimizer_config["lr"]).t()
                logger.debug("Weights update completed.")

                # calculate and update the progress of the training
                self.progress_calculator.cal_custom_progress(epoch, batch_idx+1)

            train_loss = loss_func(
                torch.tensor(training_pred_prob_list, dtype=torch.float32),
                torch.tensor(training_y_list, dtype=torch.float32)
            ).detach().item()
            self._calc_metrics(np.array(training_y_list, dtype=float), np.array(
                training_pred_prob_list), epoch)

            # Validation step should be added here.
            cm = np.zeros((2, 2))
            pred_prob_list, y_list = [], []
            for batch_idx, (x_batch, y_batch, _) in enumerate(self.val_dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred_label_trainer = self.model(x_batch)
                pred_trainer_list = broadcast_channel.collect()
                if BLOCKCHAIN:
                    logger.debug(
                        f"Collect pred, SHA256: {hashlib.sha256(pickle.dumps(pred_trainer_list)).hexdigest()}")

                # Add predictions.
                pred_total = torch.clone(pred_label_trainer)
                for pred_trainer in pred_trainer_list:
                    pred_total += pred_trainer

                pred_total = torch.sigmoid(pred_total)
                pred_prob_list += torch.squeeze(pred_total, dim=-1).tolist()
                y_list += torch.squeeze(y_batch, dim=-1).tolist()
                pred_total = (pred_total > 0.5).float()
                cm += confusion_matrix(y_true=y_batch.detach().numpy(),
                                       y_pred=pred_total.detach().numpy())

            metric = self._calc_metrics(np.array(y_list, dtype=float), np.array(pred_prob_list),
                                        epoch, stage="val")
            val_loss = loss_func(
                torch.tensor(pred_prob_list, dtype=torch.float32),
                torch.tensor(y_list, dtype=torch.float32)
            ).detach().item()

            try:
                # loss_file = self.train_conf['output']['plot_loss']['name']
                logger.info(f"Writing loss for epoch {epoch}")
                self._write_loss(train_loss, val_loss, epoch)
            except Exception:
                pass

            if self.early_stopping_config["patience"] > 0:
                early_stop_flag, save_flag = self.es(metric)
            else:
                early_stop_flag, save_flag = False, True

            if save_flag:
                self.best_model = copy.deepcopy(self.model)
                self.best_prediction_train = copy.deepcopy(
                    training_pred_prob_list)
                self.best_prediction_val = copy.deepcopy(
                    np.array(pred_prob_list))

            early_stop = [early_stop_flag, save_flag,
                          self.early_stopping_config["patience"]]
            broadcast_channel.broadcast(early_stop,
                                        use_pickle=True)
            if BLOCKCHAIN:
                logger.debug(
                    f"Broadcast early stop flag, SHA256: {hashlib.sha256(pickle.dumps(early_stop))}")

            if early_stop_flag:
                break
                # self.dump_as_proto(save_dir=self.save_dir, model_name=self.save_model_name,
                #                     state_dict=self.best_model.state_dict(), final=True)
                # # if self.save_probabilities:
                # self._save_prob(best_model=self.best_model, channel=broadcast_channel)
                # return None

            if self.save_frequency > 0 and epoch % self.save_frequency == 0:
                if self.save_model_name.split(".")[-1] == "pmodel":
                    self.dump_as_proto(
                        save_dir=self.save_dir,
                        model_name=self.save_model_name,
                        state_dict=self.model.state_dict(),
                        epoch=epoch,
                    )
                else:
                    ModelIO.save_torch_model(
                        state_dict=self.model.state_dict(), 
                        save_dir=self.save_dir, 
                        model_name=self.save_model_name,
                        epoch=epoch,
                    )

                if self.save_onnx_model_name:
                    ModelIO.save_torch_onnx(
                        model=self.model,
                        input_dim=(self.data_dim,),
                        save_dir=self.save_dir,
                        model_name=self.save_onnx_model_name,
                        epoch=epoch,
                    )

        if self.early_stopping_config["patience"] <= 0:
            self.best_model = copy.deepcopy(self.model)
            self.best_prediction_train = copy.deepcopy(training_pred_prob_list)
            self.best_prediction_val = copy.deepcopy(np.array(pred_prob_list))

        self.save(y_list, training_y_list)
        # if self.save_probabilities:
        self._save_prob(best_model=self.best_model, channel=broadcast_channel)

        self._save_feature_importance(broadcast_channel)

        # prepare data for writing
        train_label = np.array(training_y_list, dtype=float)
        train_y_pred = np.array(training_pred_prob_list, dtype=float)
        val_label = np.array(y_list, dtype=float)
        val_y_pred = np.array(pred_prob_list, dtype=float)

        # write roc data
        logger.info("Writing roc data...")
        self._write_roc_data(train_label, train_y_pred,
                             val_label, val_y_pred)

        # write ks data
        logger.info("Writing ks data...")
        self._write_ks_data(train_label, train_y_pred,
                            val_label, val_y_pred)

        # write lift and gain
        logger.info("Writing lift and gain data...")
        self._write_lift_gain_data(
            train_label, train_y_pred, val_label, val_y_pred)

        # write pr curve
        logger.info("Writing pr curve data")
        self._write_pr_data(
            train_label, train_y_pred, val_label, val_y_pred)

        # write feature importance
        logger.info("Writing feature importance data")
        logger.info("Self importances: {}".format(self.feature_importances_))
        self._write_feature_importance()

        ProgressCalculator.finish_progress()

    def save(self, y_list, training_y_list=None):
        save_model_config(stage_model_config=self.export_conf,
                          save_path=Path(self.save_dir))

        if not os.path.exists(self.evaluation_path):
            os.makedirs(self.evaluation_path)

        # dump out ks plot
        suggest_threshold = 0.5
        if "ks" in self.metric_config or "auc_ks" in self.metric_config:
            tc = ThresholdCutter(os.path.join(
                self.save_dir, self.output.get("ks_plot_val")["name"]))
            # tc.cut_by_value(np.array(y_list, dtype=float),
            #                 self.best_prediction_val)
            # suggest_threshold = tc.bst_threshold
            # tc.save()
            if self.interaction_params.get("echo_training_metrics"):
                tc = ThresholdCutter(os.path.join(
                    self.save_dir, self.output.get("ks_plot_val")["name"]))
                # tc.cut_by_value(
                #     np.array(training_y_list, dtype=float), self.best_prediction_train)
                # tc.save()
        if self.save_model_name:
            if self.save_model_name.split(".")[-1] == "pmodel":
                self.dump_as_proto(
                    save_dir=self.save_dir,
                    model_name=self.save_model_name,
                    state_dict=self.best_model.state_dict(),
                    final=True,
                    suggest_threshold=suggest_threshold
                )
            else:
                ModelIO.save_torch_model(
                    state_dict=self.best_model.state_dict(), 
                    save_dir=self.save_dir, 
                    model_name=self.save_model_name,
                    meta_dict={"suggest_threshold": suggest_threshold}
                )

        if self.save_onnx_model_name:
            ModelIO.save_torch_onnx(
                model=self.best_model,
                input_dim=(self.data_dim,),
                save_dir=self.save_dir,
                model_name=self.save_onnx_model_name,
            )

    def _save_feature_importance(self, channel):
        res = {"owner_id": [], "fid": [], "importance": []}
        other_weight_list = channel.collect()
        if BLOCKCHAIN:
            logger.debug(
                f"Collect weight list, SHA256: {hashlib.sha256(pickle.dumps(other_weight_list)).hexdigest()}")
        for (owner_id, weights, f_names) in other_weight_list:
            for fid, weight in enumerate(weights):
                res["owner_id"].append(owner_id)
                # res["fid"].append(fid)
                res["fid"].append(f_names[fid])
                res["importance"].append(float(weight))
        for fid, weight in enumerate(self.best_model.state_dict()["linear.weight"][0]):
            # res["owner_id"].append(FedNode.node_id)
            res["owner_id"].append(FedNode.node_name)
            # res["fid"].append(fid)
            f_name = self.train_f_names[fid]
            res["fid"].append(f_name)
            res["importance"].append(float(weight))
        res = pd.DataFrame(res).sort_values(
            by="importance", key=lambda col: np.abs(col), ascending=False)
        res.to_csv(
            # Path(self.save_dir, "feature_importances.csv"), header=True, index=False, float_format="%.6g"
            Path(self.save_dir, self.output["feature_importance"]["name"]), header=True, index=False,
            float_format="%.6g"
        )

        # prepare feature_importances_ attribute
        feature_importances_ = {}
        for _, row in res.iterrows():
            feature_importances_[(row['owner_id'], row['fid'])] = row['importance']
                
        self.feature_importances_ = feature_importances_

    def _save_prob(self, best_model, channel):
        if self.interaction_params.get("write_training_prediction"):
            train_prob_list, train_label_list, train_id_list = [], [], []
            for batch_idx, (x_batch, y_batch, id_batch) in enumerate(self.train_dataloader):
                x_batch, y_batch, id_batch = x_batch.to(self.device), y_batch.to(
                    self.device), id_batch.to(self.device)
                pred_label_trainer = best_model(x_batch)
                pred_trainer_list = channel.collect()
                if BLOCKCHAIN:
                    logger.debug(
                        f"Collect pred list, SHA256: {hashlib.sha256(pickle.dumps(pred_trainer_list)).hexdigest()}")
                pred_total = torch.clone(pred_label_trainer)
                for pred_trainer in pred_trainer_list:
                    pred_total += pred_trainer
                pred_total = torch.sigmoid(pred_total)
                train_id_list += torch.squeeze(id_batch, dim=-1).tolist()
                train_label_list += torch.squeeze(y_batch, dim=-1).tolist()
                train_prob_list += torch.squeeze(pred_total, dim=-1).tolist()
            self._write_prediction(
                train_label_list, train_prob_list, train_id_list, final=True)

        if self.interaction_params.get("write_validation_prediction"):
            val_prob_list, val_label_list, val_id_list = [], [], []
            for batch_idx, (x_batch, y_batch, id_batch) in enumerate(self.val_dataloader):
                x_batch, y_batch, id_batch = x_batch.to(self.device), y_batch.to(
                    self.device), id_batch.to(self.device)
                pred_label_trainer = best_model(x_batch)
                pred_trainer_list = channel.collect()
                if BLOCKCHAIN:
                    logger.debug(
                        f"Collect pred list, SHA256: {hashlib.sha256(pickle.dumps(pred_trainer_list)).hexdigest()}")
                pred_total = torch.clone(pred_label_trainer)
                for pred_trainer in pred_trainer_list:
                    pred_total += pred_trainer
                pred_total = torch.sigmoid(pred_total)
                val_id_list += torch.squeeze(id_batch, dim=-1).tolist()
                val_label_list += torch.squeeze(y_batch, dim=-1).tolist()
                val_prob_list += torch.squeeze(pred_total, dim=-1).tolist()
            self._write_prediction(
                val_label_list, val_prob_list, val_id_list, stage="val", final=True)

    def check_data(self):
        dim_channel = BroadcastChannel(name="check_data_com")
        n = self.data_dim
        dims = dim_channel.collect()
        if BLOCKCHAIN:
            logger.debug(
                f"Collect dim, SHA256: {hashlib.sha256(pickle.dumps(dims)).hexdigest()}")
        for dim in dims:
            n += dim
        if n <= 0:
            raise ValueError("Number of the feature is zero. Stop training.")
