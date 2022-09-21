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


from algorithm.core.horizontal.template.tensorflow.fedavg.label_trainer import FedAvgLabelTrainer
from common.utils.logger import logger
from .common import Common
from tqdm import tqdm
import tensorflow as tf

from python.service.fed_node import FedNode

class HorizontalBertLabelTrainer(Common, FedAvgLabelTrainer):
    def __init__(self, train_conf: dict):
        FedAvgLabelTrainer.__init__(self, train_conf)
        
    def train_loop(self):
        train_loss = 0
    
        loss_func = list(self.loss_func.values())[0]
        optimizer = list(self.optimizer.values())[0]
        
        for idx, (input_ids, token_type_ids, attention_masks, labels) in enumerate(tqdm(self.train_dataloader)):
            with tf.GradientTape() as tape:
                _,_,prob = self.model(input_ids, token_type_ids, attention_masks, labels)
                loss = loss_func(labels, prob)
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                train_loss += float(loss)

        train_loss /= len(self.train_dataloader)
        self.context["train_loss"] = train_loss
        logger.info(f"Train loss: {train_loss}")
