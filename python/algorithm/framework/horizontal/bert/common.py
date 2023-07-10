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
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from python.algorithm.model.bert import BertForSst2Torch
from common.utils.logger import logger
from algorithm.core.horizontal.template.torch.base import BaseTrainer
from common.utils.config_sync import ConfigSynchronizer
from common.checker.x_types import All
from common.evaluation.metrics import CommonMetrics


class Common(BaseTrainer):
    def __init__(self, train_conf: dict) -> None:
        sync_rule = {
            "model_info": All(),
            "train_info": {
                "interaction_params": All(),
                "train_params": {
                    "global_epoch": All(),
                    "aggregation": All(),
                    "encryption": All(),
                    "optimizer": All(),
                    "lr_scheduler": All(),
                    "lossfunc": All(),
                    "metric": All(),
                    "early_stopping": All()
                }
            }
        }
        train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
        super().__init__(train_conf)

    def _set_model(self) -> nn.Module:
        model_config = self.common_config.model_info.get("config")
        model = BertForSst2Torch(**model_config)
        return model

    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None
        conf = input_dataset[0]
        path = os.path.join(conf['path'], conf['name'])
        raw_data = pd.read_csv(path, sep='\t')
        data = raw_data["sentence"].values, raw_data["label"].values
        return data

    def _encode_examples(self, data, tokenizer, max_length=512):
        input_ids, token_type_ids, attention_masks, labels = [], [], [], []
        for feature, label in zip(*data):
            bert_input = tokenizer.encode_plus(feature,
                                               add_special_tokens=True,
                                               max_length=max_length,
                                               padding='max_length',
                                               return_token_type_ids=True,
                                               return_attention_mask=True)

            input_ids.append(bert_input['input_ids'])
            token_type_ids.append(bert_input['token_type_ids'])
            attention_masks.append(bert_input['attention_mask'])
            labels.append(label)

        return TensorDataset(torch.tensor(input_ids), torch.tensor(token_type_ids),
                             torch.tensor(attention_masks), torch.tensor(labels))

    def _set_train_dataloader(self):
        train_dataset = None
        train_dataloader = None
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_data = self._read_data(self.common_config.input_trainset)

        if train_data is not None:
            train_dataset = self._encode_examples(train_data, tokenizer)
            batch_size = self.common_config.train_params.get(
                "train_batch_size")
            train_dataloader = DataLoader(
                train_dataset, batch_size, shuffle=True)
        return train_dataloader

    def _set_val_dataloader(self):
        val_dataset = None
        val_dataloader = None
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        val_data = self._read_data(self.common_config.input_valset)
        if val_data is not None:
            val_dataset = self._encode_examples(val_data, tokenizer)
            batch_size = self.common_config.train_params.get("val_batch_size")
            val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
        return val_dataloader

    def val_loop(self, dataset_type: str = "val", context: dict = {}):
        self.model.eval()
        val_loss = 0
        val_predicts = []
        labels = []

        lossfunc_name = list(self.lossfunc.keys())[0]
        lossfunc = list(self.lossfunc.values())[0]

        if dataset_type == "val":
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for batch_id, (input_ids, token_type_ids, attention_masks, label) in enumerate(dataloader):
            _, _, pred = self.model(
                input_ids, token_type_ids, attention_masks, label)
            loss = lossfunc(pred, label)
            val_predicts.append(pred.detach().cpu().squeeze(-1).numpy())
            val_loss += loss.item()

            labels.append(label.cpu().squeeze(-1).numpy())

        val_loss /= len(dataloader)
        labels: np.ndarray = np.concatenate(labels, axis=0)
        val_predicts: np.ndarray = np.concatenate(val_predicts, axis=0)
        if len(val_predicts.shape) == 1:
            val_predicts = np.array(val_predicts > 0.5, dtype=np.int32)
        elif len(val_predicts.shape) == 2:
            val_predicts = val_predicts.argmax(axis=-1)

        metrics_output = CommonMetrics._calc_metrics(
            metrics=self.metrics,
            labels=labels,
            val_predicts=val_predicts,
            lossfunc_name=lossfunc_name,
            loss=val_loss,
            dataset_type=dataset_type
        )

        global_epoch = self.context["g_epoch"]
        if dataset_type == "val":
            local_epoch = None
        elif dataset_type == "train":
            local_epoch = self.context["l_epoch"]

        CommonMetrics.save_metric_csv(
            metrics_output=metrics_output, 
            output_config=self.common_config.output, 
            global_epoch=global_epoch, 
            local_epoch=local_epoch, 
            dataset_type=dataset_type,
        )

        early_stop_flag = self.context["early_stop_flag"]
        if (self.common_config.save_frequency > 0) & \
            (dataset_type == "val") & (self.earlystopping.patience > 0):
            early_stop_flag = self.earlystopping(metrics_output, global_epoch)
            if early_stop_flag:
                # find the saved epoch closest to the best epoch
                best_epoch = self.earlystopping.best_epoch
                closest_epoch = round(best_epoch / self.common_config.save_frequency) * \
                    self.common_config.save_frequency
                closest_epoch -= self.common_config.save_frequency \
                    if closest_epoch > global_epoch else 0
                self.context["early_stop_flag"] = True
                self.context["early_stop_epoch"] = closest_epoch
                