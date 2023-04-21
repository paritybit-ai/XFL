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
import tenseal as ts

from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.utils.logger import logger
from service.fed_config import FedConfig


class VerticalPoissonRegressionAssistTrainer(object):
    def __init__(self, *args, **kwargs):
        """[summary]
        assist_trainer
        """
        self.broadcast_channel = BroadcastChannel(name="Public keys", root_id=FedConfig.get_assist_trainer())
        self.dual_channels = {"gradients_loss": {}}
        self.party_id_list = FedConfig.get_label_trainer() + FedConfig.get_trainer()
        for party_id in self.party_id_list:
            self.dual_channels["gradients_loss"][party_id] = DualChannel(name="gradients_loss_" + party_id,
                                                                         ids=[FedConfig.node_id, party_id])
        self.batch_num = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
        self.global_epoch = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
        self.batch_size = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
        self.encryption_config = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
        self.encryption_method = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
        self.private_context = None
        self.public_context = None
        # send encryption key to all parties
        if self.encryption_method == "ckks":
            self.private_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.encryption_config[self.encryption_method]["poly_modulus_degree"],
                coeff_mod_bit_sizes=self.encryption_config[self.encryption_method]["coeff_mod_bit_sizes"]
            )
            self.private_context.generate_galois_keys()
            self.private_context.generate_relin_keys()
            self.private_context.global_scale = 1 << self.encryption_config[self.encryption_method][
                "global_scale_bit_size"]

            serialized_public_context = self.private_context.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True
            )
            logger.info("Broadcast ckks public keys.")
            self.public_context_ser = serialized_public_context
            self.broadcast_channel.broadcast(self.public_context_ser, use_pickle=False)
            logger.info("Broadcast completed.")
        elif self.encryption_method == "paillier":
            self.num_cores = -1 if self.encryption_config[self.encryption_method]["parallelize_on"] else 1
            self.private_context = Paillier.context(self.encryption_config[self.encryption_method]["key_bit_size"],
                                                    djn_on=self.encryption_config[self.encryption_method]["djn_on"])
            logger.info("Broadcast paillier public keys.")
            self.public_context_ser = self.private_context.to_public().serialize()
            self.broadcast_channel.broadcast(self.public_context_ser, use_pickle=False)
            logger.info("Broadcast completed.")
        elif self.encryption_method == "plain":
            pass
        else:
            raise ValueError(f"Encryption method {self.encryption_method} not supported! Valid methods are 'paillier', "
                             f"'ckks', 'plain'.")

    def fit(self):
        """ train model
        Model parameters need to be updated before fitting.
        """
        # send encryption key to all parties
        if self.encryption_method in ["ckks", "paillier"]:
            logger.info("Broadcast ckks public keys.")
            self.broadcast_channel.broadcast(self.public_context_ser, use_pickle=False)
            logger.info("Broadcast completed.")

        # train
        for epoch in range(1, self.global_epoch + 1):
            for batch_idx in range(self.batch_num):
                # receive and decrypt total encrypted loss and send to label_trainer
                logger.info("Receive and decrypted total loss and send back to label_trainer.")
                if self.encryption_method == "ckks":
                    enc_loss_batch = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv(
                        use_pickle=False)
                    decrypted_loss_batch = ts.ckks_vector_from(self.private_context, enc_loss_batch).decrypt()
                elif self.encryption_method == "paillier":
                    enc_loss_batch = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv(
                        use_pickle=False)
                    decrypted_loss_batch = Paillier.decrypt(self.private_context, Paillier.ciphertext_from(
                        None, enc_loss_batch), dtype='float', num_cores=self.num_cores)
                elif self.encryption_method == "plain":
                    enc_loss_batch = self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].recv()
                    decrypted_loss_batch = enc_loss_batch
                decrypted_loss_batch = np.sum(decrypted_loss_batch)
                self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].send(decrypted_loss_batch)
                logger.info(
                    "Loss of {} batch {} epoch is {}".format(batch_idx, epoch, decrypted_loss_batch / self.batch_size))

                # receive encrypted noised gradients from other parties and decrypt and send back to other parties
                if self.encryption_method == "ckks" or self.encryption_method == "paillier":
                    # trainer
                    for party_id in FedConfig.get_trainer():
                        en_noised_gradient_trainer_w = self.dual_channels["gradients_loss"][party_id].recv(
                            use_pickle=False)
                        if self.encryption_method == "ckks":
                            noised_gradient_trainer_w = ts.ckks_vector_from(self.private_context,
                                                                            en_noised_gradient_trainer_w).decrypt()
                        elif self.encryption_method == "paillier":
                            noised_gradient_trainer_w = Paillier.decrypt(self.private_context, Paillier.ciphertext_from(
                                None, en_noised_gradient_trainer_w), dtype='float', num_cores=self.num_cores)
                        self.dual_channels["gradients_loss"][party_id].send(noised_gradient_trainer_w)
                    # label_trainer
                    en_noised_gradient_label_trainer_w = self.dual_channels["gradients_loss"][
                        FedConfig.get_label_trainer()[0]].recv(use_pickle=False)
                    en_noised_gradient_label_trainer_b = self.dual_channels["gradients_loss"][
                        FedConfig.get_label_trainer()[0]].recv(use_pickle=False)
                    if self.encryption_method == "ckks":
                        noised_gradient_label_trainer_w = ts.ckks_vector_from(
                            self.private_context, en_noised_gradient_label_trainer_w).decrypt()
                        noised_gradient_label_trainer_b = ts.ckks_vector_from(
                            self.private_context, en_noised_gradient_label_trainer_b).decrypt()
                    elif self.encryption_method == "paillier":
                        noised_gradient_label_trainer_w = Paillier.decrypt(
                            self.private_context, Paillier.ciphertext_from(None, en_noised_gradient_label_trainer_w),
                            dtype='float', num_cores=self.num_cores)
                        noised_gradient_label_trainer_b = Paillier.decrypt(
                            self.private_context, Paillier.ciphertext_from(None, en_noised_gradient_label_trainer_b),
                            dtype='float', num_cores=self.num_cores)
                    # calculate sum of gradient b
                    noised_gradient_label_trainer_b = np.sum(noised_gradient_label_trainer_b)
                    grad_send = {"noised_gradient_label_trainer_w": noised_gradient_label_trainer_w,
                                 "noised_gradient_label_trainer_b": noised_gradient_label_trainer_b}
                    self.dual_channels["gradients_loss"][FedConfig.get_label_trainer()[0]].send(grad_send)
