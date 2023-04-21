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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algorithm.core.data_io import CsvReader
from algorithm.framework.vertical.vertical_model_base import VerticalModelBase
from common.utils.logger import logger
from common.utils.model_preserver import ModelPreserver
from service.fed_config import FedConfig


class VerticalPoissonRegression(nn.Module):
    def __init__(self, input_dim: int, bias: bool = False):
        super(VerticalPoissonRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=bias)
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)


class VerticalPoissonRegressionBase(VerticalModelBase):
    def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
        """_summary_

        Args:
            train_conf (dict): _description_
            label (bool, optional): _description_. Defaults to False.
        """
        super().__init__(train_conf)
        self._parse_config()
        self.train_conf = train_conf
        self.label = label
        self.data_dim = None
        self.model = None
        self.train_dataloader, self.eval_dataloader = None, None
        if FedConfig.node_id != "assist_trainer":
            self._init_dataloader()

    def _parse_config(self) -> None:
        super()._parse_config()
        self.model_name = self.model_info.get("name")
        self.save_model_name = self.output.get("model", {}).get("name")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.global_epoch = self.train_params.get("global_epoch")
        self.batch_size = self.train_params.get("batch_size")
        self.encryption_config = self.train_params.get("encryption")
        self.optimizer_config = self.train_params.get("optimizer")
        self.pretrain_model_path = self.input.get("pretrained_model", {}).get("path")
        self.random_seed = self.train_params.get("random_seed", None)
        self.early_stopping_config = self.train_params.get("early_stopping")
        self.save_frequency = self.interaction_params.get("save_frequency")

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _init_model(self, bias: bool = False) -> None:
        """
        Init poisson regression model.
        Returns: None
        """
        logger.info("Init model start.")
        self.model = VerticalPoissonRegression(input_dim=self.data_dim, bias=bias)
        # Load pretrained model if needed.
        if self.pretrain_model_path is not None and self.pretrain_model_path != "":
            checkpoint = ModelPreserver.load(os.path.join(self.pretrain_model_path, self.input.get(
                "pretrained_model").get("name", None)))
            self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("Init model completed.")

    def __load_data(self, config) -> CsvReader:
        config = config[0]
        if config["type"] == "csv":
            data_reader = CsvReader(path=os.path.join(config["path"], config["name"]), has_id=config["has_id"],
                                    has_label=config["has_label"])
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return data_reader

    def _init_data(self) -> None:
        if len(self.input_trainset) > 0:
            data: CsvReader = self.__load_data(self.input_trainset)
            self.train = data.features()
            self.train_label = data.label()
            self.train_ids = list(range(len(data.ids)))
        else:
            raise NotImplementedError("Trainset was not configured.")
        if self.label:
            assert len(self.train) == len(self.train_label)

        if len(self.input_valset) > 0:
            data: CsvReader = self.__load_data(self.input_valset)
            self.val = data.features()
            self.val_label = data.label()
            self.val_ids = list(range(len(data.ids)))
        if self.label:
            assert len(self.val) == len(self.val_label)

    def _init_dataloader(self) -> None:
        """
        Load raw data.
        Returns:

        """
        logger.info("Dataloader initiation start.")
        self._init_data()
        if self.label:
            self.train_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(self.train, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(self.train_label), dim=-1),
                                      torch.unsqueeze(torch.tensor(self.train_ids), dim=-1)),
                batch_size=self.batch_size, shuffle=True
            )
            self.val_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(self.val, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(self.val_label), dim=-1),
                                      torch.unsqueeze(torch.tensor(self.val_ids), dim=-1)),
                batch_size=self.batch_size, shuffle=False
            )
            self.data_dim = torch.tensor(self.train).shape[-1]
            logger.info("Train data shape: {}.".format(list(torch.tensor(self.train).shape)))
        else:
            self.train_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(self.train, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(self.train_ids), dim=-1)),
                batch_size=self.batch_size, shuffle=True
            )
            self.val_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(self.val, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(self.val_ids), dim=-1)),
                batch_size=self.batch_size, shuffle=False
            )
            self.data_dim = torch.tensor(self.train).shape[-1]
            logger.info("Train data shape: {}.".format(list(torch.tensor(self.train).shape)))

        logger.info("Dataloader initiation completed.")
