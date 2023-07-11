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
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

from algorithm.core.data_io import CsvReader
from algorithm.model.horizontal_k_means import HorizontalKMeans
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
                }
            }
        }
        train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
        super().__init__(train_conf)
    
    def _set_model(self) -> nn.Module:
        model_config = self.common_config.model_info.get("config")
        input_dim = model_config["input_dim"]
        num_clusters = model_config["num_clusters"]
        model = HorizontalKMeans(
            input_dim=input_dim, num_clusters=num_clusters)
        return model


    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None

        conf = input_dataset[0]

        if conf["type"] == "csv":
            path = os.path.join(conf['path'], conf['name'])
            logger.info(f"Data path: {os.path.abspath(path)}")
            has_label = conf["has_label"]
            has_id = conf['has_id']
            return CsvReader(path, has_id, has_label)
        else:
            return None

    def _set_train_dataloader(self):
        train_data = self._read_data(self.common_config.input_trainset)
        train_dataloader = None

        if train_data:
            train_dataset = TensorDataset(
                torch.Tensor(train_data.features()),
                torch.Tensor(train_data.label())
            )
            train_dataloader = DataLoader(train_dataset)
        return train_dataloader

    def _set_val_dataloader(self):
        val_data = self._read_data(self.common_config.input_valset)
        val_dataloader = None

        if val_data:
            val_dataset = TensorDataset(
                torch.Tensor(val_data.features()),
                torch.Tensor(val_data.label())
            )
            val_dataloader = DataLoader(val_dataset)
        return val_dataloader

    def val_loop(self, dataset_type: str = "val", context: dict = {}):
        if dataset_type == "val":
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")
        val_features, val_label = dataloader.dataset.tensors
        val_features = val_features.numpy()
        # val_bale = val_label.numpy()
        centroids = self.model.state_dict()['centroids'].numpy()
        kmeans = KMeans(
            n_clusters=centroids.shape[0], init=centroids, n_init=1, max_iter=1)
        kmeans.fit(val_features)
        kmeans.cluster_centers = centroids
        pred_labels = kmeans.predict(val_features)

        score = davies_bouldin_score(val_features, pred_labels)
        metrics_output = CommonMetrics._calc_metrics(
            metrics={},
            labels=val_label,
            val_predicts=pred_labels,
            lossfunc_name="davies_bouldin_score",
            loss=score,
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
