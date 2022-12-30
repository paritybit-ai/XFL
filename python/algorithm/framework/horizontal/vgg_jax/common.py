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
from jax import jit, random
import jax.numpy as jnp
import flax.linen as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image
from flax.core.frozen_dict import FrozenDict
from collections import OrderedDict

from algorithm.core.data_io import NpzReader
from algorithm.model.vgg_jax import vggjax
from common.utils.logger import logger


class Common():
    def _set_model(self) -> nn.Module:
        model = None
        self.init_params = None
        self.init_batch_stats = None
        self.state = None
        exmp_features = self.exmp_label if self.exmp_label is not None else self.exmp_assist
        model_config = self.model_info.get("config")
        model = vggjax(num_classes=model_config["num_classes"], layers=model_config["layers"])
        init_rng = random.PRNGKey(0)
        variables = model.init(init_rng, exmp_features, train=True)
        self.init_params, self.init_batch_stats = variables["params"], variables["batch_stats"]

        # init the state_dict and keys_dict used for aggregation
        self.state_dict = OrderedDict()
        self.keys_dict = OrderedDict()
        for key in ["params", "batch_stats"]:
            self.keys_dict[key] = OrderedDict()
            for i, j in variables[key].unfreeze().items():
                self.keys_dict[key][i] = []
                for k, v in j.items():
                    self.keys_dict[key][i].append(k)
                    self.state_dict[i+k] = np.asarray(v, dtype=np.float32)

        return model

    def state_to_state_dict(self):
        for i, j in self.state.params.unfreeze().items():
            for k, v in j.items():
                self.state_dict[i+k] = np.asarray(v, dtype=np.float32)
        for i, j in self.state.batch_stats.unfreeze().items():
            for k, v in j.items():
                self.state_dict[i+k] = np.asarray(v, dtype=np.float32)

    def state_dict_to_state(self):
        new_state = dict()
        for key in ["params", "batch_stats"]:
            new_state[key] = dict()
            for i, j in self.keys_dict[key].items():
                value_dict = dict()
                for k in j:
                    value_dict[k] = jnp.asarray(self.state_dict[i+k], dtype=np.float32)
                new_state[key][i] = value_dict
        new_state = FrozenDict(new_state)
        self.state = self.state.replace(params=new_state["params"], batch_stats=new_state["batch_stats"])
    
    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None
        conf = input_dataset[0]
        if conf["type"] == "npz":
            path = os.path.join(conf['path'], conf['name'])
            return NpzReader(path)
        else:
            return None
        
    def _set_train_dataloader(self):
        def img_collate_fn(batch):
            labels = []
            imgs = []
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                np.array
            ])
            for feature, label in batch:
                img = Image.fromarray(feature.numpy().astype(np.uint8))
                img = transform_train(img)
                img = np.transpose(img, (1, 2, 0))
                imgs.append(img) # [N, C, H, W] -> [N, H, W, C]
                labels.append(label.numpy())
            return jnp.stack(imgs, 0).astype(jnp.float32), jnp.stack(labels, 0).astype(jnp.int32)
            
        train_data = self._read_data(self.input_trainset)
        exmp_features = None
        trainset = None
        train_dataloader = None

        if train_data:
            trainset = TensorDataset(torch.tensor(train_data.features()[0:100]), torch.tensor(train_data.label()[0:100]))
            exmp_features = jnp.ones_like(jnp.stack(train_data.features()[0:2], 0))

        batch_size = self.train_params.get("batch_size", 64)
        if trainset:
            train_dataloader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=img_collate_fn)
        
        return train_dataloader, exmp_features

    def _set_val_dataloader(self):
        def img_collate_fn(batch):
            labels = []
            imgs = []
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                np.array
            ])
            for feature, label in batch:
                img = Image.fromarray(feature.numpy().astype(np.uint8))
                img = transform_test(img)
                img = np.transpose(img, (1, 2, 0))
                imgs.append(img) # [N, C, H, W] -> [N, H, W, C]
                labels.append(label.numpy())
            return jnp.stack(imgs, 0).astype(jnp.float32), jnp.stack(labels, 0).astype(jnp.int32)

        val_data = self._read_data(self.input_valset)
        exmp_features = None
        valset = None
        val_dataloader = None

        if val_data:
            valset = TensorDataset(torch.tensor(val_data.features()[0:100]), torch.tensor(val_data.label()[0:100]))
            exmp_features = jnp.ones_like(jnp.stack(val_data.features()[0:2], 0))

        batch_size = self.train_params.get("batch_size", 64)
        if valset:
            val_dataloader = DataLoader(valset, batch_size, shuffle=True, collate_fn=img_collate_fn)
        return val_dataloader, exmp_features

    def calculate_loss(self, params, batch_stats, batch, train):
        features, labels = batch
        # Run model. During training, we need to update the BatchNorm statistics.
        outputs = self.model.apply(
            {'params': params, 'batch_stats': batch_stats},
            features,
            train=train,
            mutable=['batch_stats'] if train else False
        )
        logits, new_model_state = outputs if train else (outputs, None)
        loss = self.loss_func(logits, labels).mean()
        preds = logits.argmax(axis=-1)
        return loss, (preds, new_model_state)

    def _set_jit_val_step(self):
        def val_step(batch, state):
            loss, (preds, _) = self.calculate_loss(state.params, state.batch_stats, batch, train=False)
            return loss, preds
        self.jit_val_step = jit(val_step)

    def val_loop(self, dataset_type: str = "validation", context: dict = {}):
        val_loss = 0
        val_predicts = []
        labels = []
        metric_output = {}
        
        if dataset_type in ["validation", "val"]:
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for batch_id, (feature, label) in enumerate(dataloader):
            loss, preds = self.jit_val_step((feature, label), self.state)
            
            val_predicts.append(preds)
            val_loss += loss.item()
            
            labels.append(label)
            
        val_loss /= len(dataloader)
        metric_output[self.loss_func_name] = val_loss

        val_predicts = jnp.concatenate(val_predicts, axis=0)
        labels = jnp.concatenate(labels, axis=0)

        metrics_conf: dict = self.train_params["metric_config"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](labels, val_predicts, **metrics_conf[method])
        logger.info(f"Metrics on {dataset_type} set: {metric_output}")
