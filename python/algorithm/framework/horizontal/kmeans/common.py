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
from pathlib import Path

from algorithm.core.data_io import CsvReader
from common.utils.logger import logger

from algorithm.model.horizontal_k_means import HorizontalKMeans

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger

from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans


class Common():

    def _set_model(self) -> nn.Module:
        model_config = self.model_info.get("config")
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
        train_data = self._read_data(self.input_trainset)
        train_dataloader = None

        if train_data:
            train_dataset = TensorDataset(
                torch.Tensor(train_data.features()),
                torch.Tensor(train_data.label())
            )
            train_dataloader = DataLoader(train_dataset)
        return train_dataloader

    def _set_val_dataloader(self):
        val_data = self._read_data(self.input_valset)
        val_dataloader = None

        if val_data:
            val_dataset = TensorDataset(
                torch.Tensor(val_data.features()),
                torch.Tensor(val_data.label())
            )
            val_dataloader = DataLoader(val_dataset)
        return val_dataloader

    def val_loop(self, dataset_type: str = "validation", context: dict = {}):
        val_features, val_label = self.val_dataloader.dataset.tensors
        val_features = val_features.numpy()
        val_bale = val_label.numpy()
        centroids = self.model.state_dict()['centroids'].numpy()
        kmeans = KMeans(
            n_clusters=centroids.shape[0], init=centroids, n_init=1, max_iter=1)
        kmeans.fit(val_features)
        kmeans.cluster_centers = centroids
        pred_labels = kmeans.predict(val_features)
        score = davies_bouldin_score(val_features, pred_labels)
        logger.info(f"Davies bouldin score on validation set: {score}")
