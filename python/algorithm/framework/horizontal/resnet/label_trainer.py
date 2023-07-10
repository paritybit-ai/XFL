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


from algorithm.core.horizontal.template.agg_type import register_agg_type_for_label_trainer
from common.utils.logger import logger
from .common import Common
from functools import partial


class HorizontalResnetLabelTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        agg_type = list(self.common_config.aggregation["method"].keys())[0]
        self.register_hook(
            place="after_train_loop", rank=1,
            func=partial(self.val_loop, "train"), desc="validation on trainset"
        )
        register_agg_type_for_label_trainer(self, 'torch', agg_type)
        
    def train_loop(self):
        self.model.train()
        train_loss = 0
    
        lossfunc = list(self.lossfunc.values())[0]
        optimizer = list(self.optimizer.values())[0]
        lr_scheduler = list(self.lr_scheduler.values())[0] if self.lr_scheduler.values() else None
        
        for batch, (feature, label) in enumerate(self.train_dataloader):
            pred = self.model(feature)
            loss = lossfunc(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_dataloader)
        if lr_scheduler:
            lr_scheduler.step()
        self.context["train_loss"] = train_loss
        logger.info(f"Train loss: {train_loss}")
