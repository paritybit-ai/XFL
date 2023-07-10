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
from typing import Optional
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from google.protobuf import json_format
from algorithm.framework.vertical.vertical_model_base import VerticalModelBase
from common.utils.model_io import ModelIO
from common.utils.logger import logger
from common.model.python.linear_model_pb2 import LinearModel


BLOCKCHAIN = False


class VerticalLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, bias: bool = False):
        super(VerticalLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=bias)
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)


class VerticalLogisticRegressionBase(VerticalModelBase):
    def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
        """_summary_

        Args:
            train_conf (dict): _description_
            label (bool, optional): _description_. Defaults to False.
        """
        super().__init__(train_conf)
        self._parse_config()
        self.train_conf = train_conf
        self.model_conf = train_conf["model_info"].get("config")
        self.label = label
        self.schema = None
        self.data_dim = None
        self.model = None
        self.train_dataloader, self.eval_dataloader = None, None
        self.loss_function = None
        self.metric_functions = {}
        self._init_dataloader()

    def _parse_config(self) -> None:
        super()._parse_config()
        self.model_name = self.model_info.get("name")
        self.save_model_name = self.output.get("model", {}).get("name", "")
        self.save_onnx_model_name = self.output.get("onnx_model", {}).get("name", "")

        self.evaluation_path = self.save_dir

        self.global_epoch = self.train_params.get("global_epoch")
        self.batch_size = self.train_params.get("batch_size")

        self.encryption_config = self.train_params.get("encryption")
        self.optimizer_config = self.train_params.get("optimizer")

        self.pretrain_model_path = self.input.get("pretrained_model", {}).get("path")
        self.pretrain_model_name = self.input.get("pretrained_model", {}).get("name")
        self.random_seed = self.train_params.get("random_seed")
        self.early_stopping_config = self.train_params.get("early_stopping")

        self.save_frequency = self.interaction_params.get("save_frequency")
        self.save_probabilities = self.interaction_params.get("save_probabilities")

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
            if self.pretrain_model_name.split(".")[-1] == "model":
                model_dict = ModelIO.load_torch_model(os.path.join(self.pretrain_model_path, self.pretrain_model_name))
                self.model.load_state_dict(model_dict["state_dict"])
            # elif self.pretrain_model_name.split(".")[-1] == "pmodel":
            #     checkpoint = self.load_from_proto(os.path.join(self.pretrain_model_path, self.pretrain_model_name))
            #     self.model.load_state_dict(checkpoint)
            else:
                raise NotImplementedError(
                    "Pretrained model {} does not support.".format(self.pretrain_model_name)
                )
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
        self.schema = ','.join([_ for _ in node_train_df.columns if _ not in set(["y", "id"])])

        if node_train_df.index.dtype == 'O':
            node_train_df = node_train_df.reset_index(drop=True)
        if node_val_df.index.dtype == 'O':
            node_val_df = node_val_df.reset_index(drop=True)

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
            self.train_f_names = node_val_df.columns.tolist()[1:]

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
            self.train_f_names = node_val_df.columns.tolist()
            
            self.val_dataloader = DataLoader(
                dataset=TensorDataset(torch.tensor(node_val_data, dtype=torch.float32),
                                      torch.unsqueeze(torch.tensor(node_val_id), dim=-1)),
                batch_size=self.batch_size, shuffle=False
            )
            self.data_dim = torch.tensor(node_train_data).shape[-1]
            logger.info("Data shape: {}.".format(list(torch.tensor(node_train_data).shape)))

        logger.info("Init dataloader completed.")

    # unused
    @staticmethod
    def load_from_proto(path: str):
        with open(path, 'rb') as f:
            b = f.read()
        lr = LinearModel()
        lr.ParseFromString(b)
        d = json_format.MessageToDict(lr,
                                      including_default_value_fields=True,
                                      preserving_proto_field_name=True)
        state_dict = OrderedDict()
        for k, v in d.items():
            state_dict[k] = torch.Tensor([v])
        return state_dict

    @staticmethod
    def dump_as_proto(save_dir: str,
                      model_name: str,
                      state_dict: OrderedDict,
                      epoch: int = None,
                      final: bool = False,
                      suggest_threshold: float = None
                      ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        json_dict = dict()

        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                json_dict[k.replace("linear.", "")] = v.tolist()[0]

        model_info = {"state_dict": json_dict}
        if suggest_threshold:
            model_info["suggest_threshold"] = suggest_threshold

        model_name_list = model_name.split(".")
        name_prefix, name_postfix = ".".join(model_name_list[:-1]), model_name_list[-1]

        if not final and epoch:
            model_name = name_prefix + "_epoch_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix

        model_path = os.path.join(save_dir, model_name)

        lr = LinearModel()
        json_format.ParseDict(model_info, lr)

        with open(model_path, 'wb') as f:
            f.write(lr.SerializeToString())

        logger.info("model saved as: {}.".format(model_path))
        return
