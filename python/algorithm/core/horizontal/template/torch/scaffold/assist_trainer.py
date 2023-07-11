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


from algorithm.core.horizontal.aggregation.aggregation_base import AggregationRootBase
from ..base import BaseTrainer
from service.fed_control import ProgressCalculator


class SCAFFOLDAssistTrainer:
    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer
        self.progress_calculator = ProgressCalculator(trainer.context["global_epoch_num"])

    def register(self):
        self.trainer.register_hook(
            place="before_global_epoch", rank=-1,
            func=self.trainer._load_model, desc="load pretrain model"
        )
        self.trainer.register_hook(
            place="after_local_epoch", rank=-2,
            func=self._aggregate_model, desc="aggregate local models"
        )
        self.trainer.register_hook(
            place="before_local_epoch", rank=-2,
            func=self._sync_early_stop_flag, desc="update progress bar"
        )
        self.trainer.register_hook(
            place="before_local_epoch", rank=-1,
            func=self._broadcast_model, desc="broadcast global model"
        )
        self.trainer.register_hook(
            place="after_local_epoch", rank=-1,
            func=self.progress_calculator.cal_horizontal_progress, 
            desc="update progress bar"
        )
        self.trainer.register_hook(
            place="after_global_epoch", rank=-1,
            func=self.progress_calculator.finish_progress, desc="update progress bar"
        )

    def _sync_early_stop_flag(self, context: dict):
        aggregator: AggregationRootBase = self.trainer.aggregator
        aggregator.broadcast(context["early_stop_flag"])
        return context["early_stop_flag"]
        
    def _broadcast_model(self, context: dict):
        aggregator: AggregationRootBase = self.trainer.aggregator
        aggregator.broadcast(self.trainer.model.state_dict())

    def _aggregate_model(self, context: dict):
        aggregator: AggregationRootBase = self.trainer.aggregator
        new_state_dict = aggregator.aggregate()
        if self.trainer.device != "cpu":
            self.trainer._state_dict_to_device(
                new_state_dict, self.trainer.device, inline=True)
        self.trainer.model.load_state_dict(new_state_dict)
