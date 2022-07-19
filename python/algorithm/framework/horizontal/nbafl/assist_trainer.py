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


from algorithm.core.horizontal.template.torch.fedavg.assist_trainer import FedAvgAssistTrainer
from common.utils.logger import logger
import numpy as np
from algorithm.model.logistic_regression import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from service.fed_config import FedConfig
import os
from algorithm.core.data_io import CsvReader


class HorizontalNbaflAssistTrainer(FedAvgAssistTrainer):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        self.load_model()

        self.mu = self.train_params['mu']
        self.delta = self.train_params['delta']
        self.c = np.sqrt(2 * np.log(1.25 / self.delta))
        self.epsilon = self.train_params['epsilon']

        self.sample_size_channel = {}

        # Init size channel
        for party_id in FedConfig.get_label_trainer():
            self.sample_size_channel[party_id] = DualChannel(
                name="sample_size_" + party_id,
                ids=[FedConfig.node_id, party_id]
            )

        # Get sample size
        self.register_hook(
            place="before_global_epoch", rank=2,
            func=self._get_sample_size, desc="Get sample size"
        )
        # Calculate downlink noise
        self.register_hook(
            place="before_global_epoch", rank=3,
            func=self._calc_downlink_sigma, desc="Calculate downlink noise"
        )
        # Add noise
        self.register_hook(
            place="after_local_epoch", rank=2,
            func=self._add_noise, desc="Add downlink noise"
        )
        # Validation
        self.register_hook(
            place="after_local_epoch", rank=3,
            func=self.val_loop, desc="Valdiation"
        )

    def _set_model(self):
        logger.info("Model info: {}".format(self.model_info))
        model_config = self.model_info["config"]
        assert len(model_config['layer_dim']) == len(
            model_config['activation']), "Hidden layer nums must match activation nums"
        layer_dims = [model_config['input_dim']] + model_config['layer_dim']
        layer_act = model_config['activation']
        module_list = []
        for input_dim, output_dim, activation_str in zip(layer_dims, layer_dims[1:], layer_act):
            module_list.append(
                nn.Linear(input_dim, output_dim, bias=model_config['bias']))
            activation = getattr(nn, activation_str)()
            module_list.append(activation)

        model = nn.Sequential(*module_list)

        return model

    def load_model(self):
        self._load_model({})

    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None

        conf = input_dataset[0]

        if conf["type"] == "csv":
            path = os.path.join(conf['path'], conf['name'])
            has_label = conf["has_label"]
            has_id = conf['has_id']
            return CsvReader(path, has_id, has_label)
        else:
            return None

    def _set_train_dataloader(self):
        train_data = self._read_data(self.input_trainset)
        trainset = None
        train_dataloader = None

        if train_data:
            trainset = TensorDataset(torch.tensor(train_data.features(), dtype=torch.float32).to(self.device),
                                     torch.tensor(train_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(self.device))

        batch_size = self.train_params.get("batch_size", 64)
        if trainset:
            train_dataloader = DataLoader(trainset, batch_size, shuffle=True)
        return train_dataloader

    def _set_val_dataloader(self):
        val_data = self._read_data(self.input_valset)
        valset = None
        val_dataloader = None

        if val_data:
            valset = TensorDataset(torch.tensor(val_data.features(), dtype=torch.float32).to(self.device),
                                   torch.tensor(val_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(self.device))

        batch_size = self.train_params.get("batch_size", 64)
        if valset:
            val_dataloader = DataLoader(valset, batch_size, shuffle=True)
        return val_dataloader

    def _calc_downlink_sigma(self, context):
        logger.info("Calculating downlink sigma")
        if self.train_params['global_epoch'] > self.train_params['num_client'] * np.sqrt(self.train_params['num_client']):
            sigma_d = (
                2 * self.train_params['C'] * self.c * np.sqrt(
                    self.train_params['global_epoch'] ** 2 - np.power(self.train_params['num_client'], 3))
                / (self.min_sample_num * self.train_params['num_client'] * self.train_params['epsilon'])
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

    def val_loop(self, context):
        self.model.eval()
        val_loss = 0
        pred_list = []
        label_list = []
        metric_output = {}

        loss_func = next(iter(self.loss_func.items()))[1]

        with torch.no_grad():
            for batch_idx, (feature, label) in enumerate(self.val_dataloader):
                pred = self.model(feature)
                loss = loss_func(pred, label)
                val_loss += loss
                pred_list.append(pred.detach().cpu().squeeze(-1).numpy())
                label_list.append(label.detach().cpu().squeeze(-1).numpy())

        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        if len(preds.shape) == 1:
            preds = np.array(preds > 0.5, dtype=np.int32)
        elif len(preds.shape) == 2:
            preds = preds.argmax(axis=-1)

        logger.info("Validation loss: {}".format(
            val_loss/len(self.val_dataloader.dataset)))
        metrics_conf: dict = self.train_params["metric_config"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](
                labels, preds, **metrics_conf[method])
            logger.info("Validation {}: {}".format(
                method, metric_output[method]))
