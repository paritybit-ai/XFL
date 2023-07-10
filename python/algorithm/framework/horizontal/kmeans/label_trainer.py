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


import torch
from torch import nn
from sklearn.cluster import KMeans
from algorithm.core.horizontal.template.agg_type import register_agg_type_for_label_trainer
from common.utils.logger import logger
from .common import Common
from functools import partial


class HorizontalKmeansLabelTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        self.register_hook(
            place="after_train_loop", rank=1,
            func=partial(self.val_loop, "train"), desc="validation on trainset"
        )
        register_agg_type_for_label_trainer(self, "torch", "fedavg")
        # logger.info("Label trainer initialized")

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
