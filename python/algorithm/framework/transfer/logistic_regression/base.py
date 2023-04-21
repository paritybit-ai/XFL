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

from algorithm.framework.transfer.transfer_model_base import TransferModelBase
from common.utils.logger import logger
from common.utils.model_preserver import ModelPreserver


class TransferLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class TransferLogisticRegressionBase(TransferModelBase):
    def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
        """_summary_

        Args:
            train_conf (dict): _description_
            label (bool, optional): _description_. Defaults to False.
        """
        super().__init__(train_conf)
        self._parse_config()
        self.label = label
        self.model = None
        self.phi = None # phi will be saved in the checkpoint of label_trainer
        self.overlap_y, self.non_overlap_y = None, None
        self.overlap_train_dataloader, self.non_overlap_train_dataloader = None, None
        self.eval_dataloader = None
        self.metric_functions = {}
        self._set_train_dataloader()
        self._set_val_dataloader()

    def _init_model(self, bias: bool = False) -> None:
        """
        Init logistic regression model.
        Returns: None
        """
        logger.info("Init model start.")
        self.model = TransferLogisticRegression(
            input_dim=self.num_features, output_dim=self.hidden_features, bias=bias
        )
        # Load pretrained model if needed.
        if self.pretrain_model_path is not None and self.pretrain_model_path != "":
            checkpoint = ModelPreserver.load(
                os.path.join(self.pretrain_model_path, self.input.get("pretrained_model").get("name")))
            state_dict = checkpoint["state_dict"]
            if "phi" in state_dict.keys():
                self.phi = state_dict.pop("phi")
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        logger.info("Init model completed.")

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
        train_data, overlap_index = self._read_data(self.input_trainset)
        self.sample_num = train_data.shape[0]
        overlap_train_data = train_data.loc[overlap_index]
    
        if self.label:
            non_overlap_index = np.array([])
            for i in train_data.index:
                if i not in overlap_index:
                    non_overlap_index = np.append(non_overlap_index, i)
            non_overlap_train_data = train_data.loc[non_overlap_index]

            # init overlap_y and non_overlap_y
            self.overlap_y = torch.tensor(overlap_train_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(1)
            self.non_overlap_y = torch.tensor(non_overlap_train_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(1)

            # init train_dataloader
            overlap_x = torch.tensor(overlap_train_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            overlap_trainset = TensorDataset(overlap_x, self.overlap_y)
            self.overlap_train_dataloader = DataLoader(overlap_trainset, batch_size=self.batch_size, shuffle=False)

            non_overlap_x = torch.tensor(non_overlap_train_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            non_overlap_trainset = TensorDataset(non_overlap_x, self.non_overlap_y)
            self.non_overlap_train_dataloader = DataLoader(non_overlap_trainset, batch_size=self.batch_size, shuffle=False)

        else:
            # init train_dataloader
            overlap_x = torch.tensor(overlap_train_data.to_numpy(), dtype=torch.float32)
            overlap_trainset = TensorDataset(overlap_x)
            self.overlap_train_dataloader = DataLoader(overlap_trainset, batch_size=self.batch_size, shuffle=False)

    def _set_val_dataloader(self):
        val_data = self._read_data(self.input_valset, is_train=False)
    
        if self.label:
            # init val_dataloader
            labels = torch.tensor(val_data.iloc[:, 0].to_numpy(), dtype=torch.float32).unsqueeze(dim=-1)
            valset = TensorDataset(labels)
            self.val_dataloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        else:
            # init val_dataloader
            features = torch.tensor(val_data.to_numpy(), dtype=torch.float32)
            valset = TensorDataset(features)
            self.val_dataloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)