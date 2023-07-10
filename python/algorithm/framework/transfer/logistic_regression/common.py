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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from ..base import BaseTrainer
from common.utils.logger import logger
from common.utils.model_io import ModelIO
from common.checker.x_types import All
from common.utils.config_sync import ConfigSynchronizer


class TransferLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class Common(BaseTrainer):
    def __init__(self, train_conf: dict, label: bool = False):
        sync_rule = {
            "model_info": {
                "config": All()
            },
            "train_info": {
                "interaction_params": All(),
                "train_params": All()
            }
        }
        train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
        self.label = label
        super().__init__(train_conf)
        self._parse_config()
    
    def _parse_config(self) -> None:
        self.save_dir = Path(self.common_config.output.get("path", ""))

        # interaction_params
        self.model_name = self.common_config.model_info.get("name")
        self.save_model_name = self.common_config.output.get("model", {}).get("name", {})
        self.pretrain_model_path = self.common_config.input.get("pretrained_model", {}).get("path")
        self.pretrain_model_name = self.common_config.input.get("pretrained_model", {}).get("name")
        self.model_conf = self.common_config.model_info.get("config", {})
        self.num_features = self.model_conf.get("num_features")
        self.hidden_features = self.model_conf.get("hidden_features")
        self.constant_k = 1 / self.hidden_features
        self.alpha = self.model_conf.get("alpha")
        self.bias = self.model_conf.get("bias", False)

        self.global_epoch = self.common_config.train_params.get("global_epoch")
        self.local_epoch = self.common_config.train_params.get("local_epoch")
        self.train_batch_size = self.common_config.train_params.get("train_batch_size")
        self.val_batch_size = self.common_config.train_params.get("val_batch_size")

    def _set_model(self) -> None:
        """
        Init logistic regression model.
        Returns: None
        """
        logger.info("Init model start.")
        self.phi = None # phi will be saved in the model_info of label_trainer
        model = TransferLogisticRegression(
            input_dim=self.num_features, output_dim=self.hidden_features, bias=self.bias
        )
        # Load pretrained model if needed.
        if self.pretrain_model_path is not None and self.pretrain_model_path != "":
            model_info = ModelIO.load_torch_model(
                os.path.join(self.pretrain_model_path, self.pretrain_model_name))
            state_dict = model_info["state_dict"]
            if "phi" in state_dict.keys():
                self.phi = model_info["phi"]
            model.load_state_dict(state_dict)

        model = model.to(self.device)
        logger.info("Init model completed.")
        return model

    def _read_data(self, input_dataset, is_train=True):
        if len(input_dataset) == 0:
            return None
        
        conf = input_dataset[0]
        if conf["type"] == "csv":
            path = os.path.join(conf['path'], conf['name'])
            has_id = conf['has_id']
            index_col = 0 if has_id else False
            train_data = pd.read_csv(path, index_col=index_col)
            if is_train:
                index_name = "overlap_index.npy"
                index_path = os.path.join(conf['path'], index_name)
                overlap_index = np.load(index_path)
                return train_data, overlap_index
            else:
                return train_data
        else:
            raise NotImplementedError(
                    "Dataset load method {} does not Implemented.".format(conf["type"])
                )
    
    def _set_train_dataloader(self):
        self.overlap_y, self.non_overlap_y = None, None
        self.overlap_train_dataloader, self.non_overlap_train_dataloader = None, None
        train_data, overlap_index = self._read_data(self.common_config.input_trainset)
        self.sample_num = train_data.shape[0]
        overlap_train_data = train_data.loc[overlap_index]
    
        if self.label:
            non_overlap_index = np.array([])
            for i in train_data.index:
                if i not in overlap_index:
                    non_overlap_index = np.append(non_overlap_index, i)
            non_overlap_train_data = train_data.loc[non_overlap_index]
            if len(non_overlap_index) == 0:
                raise ValueError("There is no non-overlap data in the trainset. If non_overlap_index is empty, there is no need to use transfer learning")

            # init overlap_y and non_overlap_y
            self.overlap_y = torch.tensor(
                overlap_train_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(1)
            self.non_overlap_y = torch.tensor(
                non_overlap_train_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(1)

            # init train_dataloader
            overlap_x = torch.tensor(
                overlap_train_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            overlap_trainset = TensorDataset(overlap_x, self.overlap_y)
            self.overlap_train_dataloader = DataLoader(
                overlap_trainset, batch_size=self.train_batch_size, shuffle=False)

            non_overlap_x = torch.tensor(
                non_overlap_train_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            non_overlap_trainset = TensorDataset(non_overlap_x, self.non_overlap_y)
            self.non_overlap_train_dataloader = DataLoader(
                non_overlap_trainset, batch_size=self.train_batch_size, shuffle=False)

        else:
            # init train_dataloader
            overlap_x = torch.tensor(overlap_train_data.to_numpy(), dtype=torch.float32)
            overlap_trainset = TensorDataset(overlap_x)
            self.overlap_train_dataloader = DataLoader(
                overlap_trainset, batch_size=self.train_batch_size, shuffle=False)

    def _set_val_dataloader(self):
        self.val_dataloader = None
        val_data = self._read_data(self.common_config.input_valset, is_train=False)
    
        if self.label:
            # init val_dataloader
            labels = torch.tensor(val_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(dim=-1)
            valset = TensorDataset(labels)
            self.val_dataloader = DataLoader(valset, batch_size=self.val_batch_size, shuffle=False)
        else:
            # init val_dataloader
            features = torch.tensor(val_data.to_numpy(), dtype=torch.float32)
            valset = TensorDataset(features)
            self.val_dataloader = DataLoader(valset, batch_size=self.val_batch_size, shuffle=False)

    