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
from pathlib import Path

from functools import partial

from algorithm.core.data_io import CsvReader
from common.utils.logger import logger

from algorithm.model.horizontal_k_means import HorizontalKMeans

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger

from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from .common import Common

from algorithm.core.horizontal.template.torch.fedavg.label_trainer import FedAvgLabelTrainer


class HorizontalKmeansLabelTrainer(Common, FedAvgLabelTrainer):
    def __init__(self, train_conf: dict):
        FedAvgLabelTrainer.__init__(self, train_conf=train_conf)
        logger.info("Label trainer initialized")

    def train_loop(self):

        # load centroids
        centroids = self.model.state_dict()['centroids'].numpy()

        # train one iter of KMeans
        kmeans_model = KMeans(
            n_clusters=centroids.shape[0],
            init=centroids,
            n_init=1,
            max_iter=10
        )
        train_features, _ = self.train_dataloader.dataset.tensors
        train_features = train_features.numpy()
        kmeans_model.fit(train_features)
        logger.info(f"K-Means score: {kmeans_model.score(train_features)}")

        # write centroids
        model_state_dict = self.model.state_dict()
        model_state_dict['centroids'] = torch.tensor(
            kmeans_model.cluster_centers_)
        # self.model.load_state_dict(model_state_dict)
        self.model.centroids = nn.Parameter(
            torch.tensor(kmeans_model.cluster_centers_))
