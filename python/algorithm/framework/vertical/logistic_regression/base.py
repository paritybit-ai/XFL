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
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algorithm.framework.vertical.vertical_model_base import VerticalModelBase
from common.utils.logger import logger
from common.utils.model_preserver import ModelPreserver


class VerticalLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, bias: bool = False):
        super(VerticalLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=bias)
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)


class VerticalLogisticRegressionBase(VerticalModelBase):
    def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
        super().__init__(train_conf)
        self._parse_config()
        self.train_conf = train_conf
        self.model_conf = train_conf["model_info"].get("config")
        self.label = label
        self.data_dim = None
        self.model = None
        self.train_dataloader, self.eval_dataloader = None, None
        self.loss_function = None
        self.metric_functions = {}
        self._init_dataloader()

    def _parse_config(self) -> None:
        super()._parse_config()
        self.model_name = self.model_info.get("name")
        self.save_model_name = self.output.get("model").get("name")

        if self.output.get("evaluation"):
            self.evaluation_path = Path(self.output["evaluation"].get("path"))
        else:
            self.evaluation_path = self.save_dir

        self.global_epoch = self.train_params.get("global_epoch")
        self.batch_size = self.train_params.get("batch_size")
        self.aggregation_config = self.train_params.get("aggregation_config")
        self.optimizer_config = self.train_params.get("optimizer_config")

        self.pretrain_model_path = self.input.get("pretrain_model").get("path")
        self.extra_config = self.train_params.get("extra_config")
        self.early_stopping_config = self.train_params.get("early_stopping")

        self.save_frequency = self.interaction_params.get("save_frequency")
        self.save_probabilities = self.interaction_params.get("save_probabilities")
        self.save_probabilities_bins_number = self.interaction_params.get("save_probabilities_bins_number")

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _init_model(self, bias: bool = False) -> None:
        """
        Init logistic regression model.
        Returns: None
        """
        logger.info("Init model start.")
        self.model = VerticalLogisticRegression(input_dim=self.data_dim, bias=bias)
        # Load pretrained model if needed.
        if self.pretrain_model_path is not None and self.pretrain_model_path != "":
            checkpoint = ModelPreserver.load(self.pretrain_model_path)
            self.model = torch.load(checkpoint["state_dict"])
            # self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("Init model completed.")

    def _init_dataloader(self) -> None:
        """
        Load raw data.
        Returns:

        """
        logger.info("Init validation dataloader start.")

        df_list = []
        # Check file exists.
        for ts in self.input_trainset:
            file_path = os.path.join(ts.get("path"), ts.get("name"))
            if not os.path.exists(file_path):
                raise FileNotFoundError("File {} cannot be found.".format(file_path))
            if ts.get("type") == "csv":
                if ts.get("has_id"):
                    df_list.append(pd.read_csv(file_path, index_col=0))
                else:
                    df_list.append(pd.read_csv(file_path))
            else:
                raise NotImplementedError(
                    "LDataset load method {} does not Implemented.".format(ts.get("type"))
                )
        node_train_df = pd.concat(df_list)

        df_list = []
        for vs in self.input_valset:
            file_path = os.path.join(vs.get("path"), vs.get("name"))
            if not os.path.exists(file_path):
                raise FileNotFoundError("File {} cannot be found.".format(file_path))
            if vs.get("type") == "csv":
                if vs.get("has_id"):
                    df_list.append(pd.read_csv(file_path, index_col=0))
                else:
                    df_list.append(pd.read_csv(file_path))
            else:
                raise NotImplementedError(
                    "Dataset load method {} does not Implemented.".format(vs.get("type"))
                )
        node_val_df = pd.concat(df_list)

        if self.label:
            # Check column y exists.
            if "y" not in node_train_df.columns:
                raise KeyError("Cannot found column y in train set.")
            if "y" not in node_val_df.columns:
                raise KeyError("Cannot found column y in val set.")

            node_train_id = node_train_df.index.to_list()
            node_train_label = node_train_df["y"].values  # .tolist()
            node_train_data = node_train_df.drop(labels=["y"], axis=1).values  # .tolist()
            assert len(node_train_label) == len(node_train_data)

            node_val_id = node_val_df.index.to_list()
            node_val_label = node_val_df["y"].values  # .tolist()
            node_val_data = node_val_df.drop(labels=["y"], axis=1).values  # .tolist()
            assert len(node_val_label) == len(node_val_data)

            self.train_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(node_train_data, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(node_train_label), dim=-1),
                                      torch.unsqueeze(torch.tensor(node_train_id), dim=-1)),
                batch_size=self.batch_size, shuffle=True
            )
            self.val_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(node_val_data, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(node_val_label), dim=-1),
                                      torch.unsqueeze(torch.tensor(node_val_id), dim=-1)),
                batch_size=self.batch_size, shuffle=False
            )
            self.data_dim = torch.tensor(node_train_data).shape[-1]
            logger.info("Data shape: {}.".format(list(torch.tensor(node_train_data).shape)))
        else:
            node_train_id = node_train_df.index.to_list()
            node_train_data = node_train_df.values.tolist()

            node_val_id = node_val_df.index.to_list()
            node_val_data = node_val_df.values.tolist()

            self.train_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(node_train_data, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(node_train_id), dim=-1)),
                batch_size=self.batch_size, shuffle=True
            )
            self.val_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(node_val_data, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(node_val_id), dim=-1)),
                batch_size=self.batch_size, shuffle=False
            )
            self.data_dim = torch.tensor(node_train_data).shape[-1]
            logger.info("Data shape: {}.".format(list(torch.tensor(node_train_data).shape)))

        logger.info("Init dataloader completed.")
