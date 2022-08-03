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


import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algorithm.core.data_io import CsvReader
from algorithm.model.logistic_regression import LogisticRegression
from common.utils.logger import logger


class Common():
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
