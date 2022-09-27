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


from algorithm.core.horizontal.template.torch.fedtype import _get_label_trainer
from common.utils.logger import logger
from .common import Common


class HorizontalLinearRegressionLabelTrainer(Common, _get_label_trainer()):
    def __init__(self, train_conf: dict):
        _get_label_trainer().__init__(self, train_conf)
        
    def train_loop(self):
        self.model.train()
        train_loss = 0
    
        loss_func = list(self.loss_func.values())[0]
        optimizer = list(self.optimizer.values())[0]
        lr_scheduler = list(self.lr_scheduler.values())[0] if self.lr_scheduler.values() else None

        for batch, (feature, label) in enumerate(self.train_dataloader):
            pred = self.model(feature)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_dataloader.dataset)
        
        if lr_scheduler:
            lr_scheduler.step()
            
        self.context["train_loss"] = train_loss
        logger.info(f"Train loss: {train_loss}")
