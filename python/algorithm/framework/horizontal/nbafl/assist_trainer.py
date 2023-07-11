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
from service.fed_config import FedConfig
from functools import partial
from common.utils.logger import logger
from .common import Common
from algorithm.core.horizontal.template.agg_type import \
    register_agg_type_for_assist_trainer


class HorizontalNbaflAssistTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        register_agg_type_for_assist_trainer(self, "torch", "fedavg")

        self.load_model()
        self.sample_size_channel = {}

        # Init size channel
        for party_id in FedConfig.get_label_trainer():
            self.sample_size_channel[party_id] = DualChannel(
                name="sample_size_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )

        # Get sample size
        self.register_hook(
            place="before_global_epoch", rank=1,
            func=self._get_sample_size, desc="Get sample size"
        )
        # Calculate downlink noise
        self.register_hook(
            place="before_global_epoch", rank=2,
            func=self._calc_downlink_sigma, desc="Calculate downlink noise"
        )
        # Add noise
        self.register_hook(
            place="after_local_epoch", rank=1,
            func=self._add_noise, desc="Add downlink noise"
        )
        # Validation
        self.register_hook(
            place="after_local_epoch", rank=2,
            func=partial(self.val_loop, "val"), desc="validation on valset"
        )
        self.register_hook(
            place="after_global_epoch", rank=1,
            func=partial(self._save_model, True), desc="save final model"
        )

    def _calc_downlink_sigma(self, context):
        logger.info("Calculating downlink sigma")
        if self.common_config.train_params['global_epoch'] > \
            self.common_config.train_params['num_client'] * \
                np.sqrt(self.common_config.train_params['num_client']):
            sigma_d = (
                2 * self.common_config.train_params['C'] * self.c * np.sqrt(
                    self.common_config.train_params['global_epoch'] ** 2 - \
                        np.power(self.common_config.train_params['num_client'], 3)) / \
                            (self.min_sample_num * \
                             self.common_config.train_params['num_client'] * \
                                self.common_config.train_params['epsilon'])
            )
        else:
            sigma_d = 0.0

        logger.info("Downlink sigma: {}".format(sigma_d))

        self.sigma_d = sigma_d

        return

    def _add_noise(self, context):
        if self.sigma_d > 0:
            noise_generator = torch.distributions.Normal(
                loc=0, scale=self.sigma_d)
            for param_data in self.model.parameters():
                param_data.data += noise_generator.sample(param_data.size())
        return

    def _get_sample_size(self, context):
        sample_nums = []
        for party_id in FedConfig.get_label_trainer():
            single_sample_size = self.sample_size_channel[party_id].recv()
            sample_nums.append(single_sample_size)
        sample_num_array = np.array(sample_nums)
        logger.info("Sample num array: {}".format(sample_num_array))
        self.min_sample_num = np.min(sample_num_array)
        return

    def train_loop(self):
        pass
