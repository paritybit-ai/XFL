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


from pathlib import Path
import torch
from functools import partial
from common.utils.config_parser import TrainConfigParser
from algorithm.core.metrics import get_metric


class TransferModelBase(TrainConfigParser):
    def __init__(self, train_conf: dict, label: bool = False):
        super().__init__(train_conf)
        self.train_conf = train_conf
        self.model_conf = train_conf["model_info"].get("config")
        self.label = label

    def _parse_config(self) -> None:
        self.save_dir = Path(self.output.get("path", ""))
        self.metric_dir = self.save_dir

        # interaction_params
        self.model_name = self.model_info.get("name")
        self.save_model_name = self.output.get("model", {}).get("name", {})
        self.pretrain_model_path = self.input.get("pretrained_model", {}).get("path")
        
        self.num_features = self.model_conf.get("num_features")
        self.hidden_features = self.model_conf.get("hidden_features")
        self.constant_k = 1 / self.hidden_features
        self.alpha = self.model_conf.get("alpha")

        self.global_epoch = self.train_params.get("global_epoch")
        self.local_epoch = self.train_params.get("local_epoch")
        self.batch_size = self.train_params.get("batch_size")

        self.shuffle_seed = self.train_params.get("shuffle_seed")
        self.random_seed = self.train_params.get("random_seed")

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    def _set_metrics(self):
        """ Define metric """
        metrics = {}
        metrics_conf: dict = self.train_params.get("metric", {})

        for k, v in metrics_conf.items():
            metric = get_metric(k)
            metrics[k] = partial(metric, **v)
        return metrics
