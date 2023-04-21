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
from algorithm.model.linear_regression import LinearRegression
from common.utils.logger import logger


class Common():
    def _set_model(self) -> nn.Module:
        model_config = self.model_info.get("config")
        model = LinearRegression(input_dim=model_config["input_dim"],
                                 bias=model_config["bias"])
        return model

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
                                     torch.tensor(train_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(
                                         self.device))

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
                                   torch.tensor(val_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(
                                       self.device))

        batch_size = self.train_params.get("batch_size", 64)
        if valset:
            val_dataloader = DataLoader(valset, batch_size, shuffle=True)
        return val_dataloader

    def val_loop(self, dataset_type: str = "validation", context: dict = {}):
        self.model.eval()
        val_loss = 0
        val_predicts = []
        labels = []
        metric_output = {}

        loss_func_name = list(self.loss_func.keys())[0]
        loss_func = list(self.loss_func.values())[0]

        if dataset_type in ["validation", "val"]:
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for batch, (feature, label) in enumerate(dataloader):
            pred = self.model(feature)
            loss = loss_func(pred, label)

            val_predicts.append(pred.detach().cpu().squeeze(-1).numpy())
            val_loss += loss.item()

            labels.append(label.cpu().squeeze(-1).numpy())

        val_loss /= len(dataloader)
        metric_output[loss_func_name] = val_loss

        val_predicts = np.concatenate(val_predicts, axis=0)
        labels = np.concatenate(labels, axis=0)
        # if len(val_predicts.shape) == 1:
        #     val_predicts = np.array(val_predicts > 0.5, dtype=np.int32)
        # elif len(val_predicts.shape) == 2:
        #     val_predicts = val_predicts.argmax(axis=-1)

        metrics_conf: dict = self.train_params["metric_config"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](labels, val_predicts, **metrics_conf[method])
        logger.info(f"Metrics on {dataset_type} set: {metric_output}")
