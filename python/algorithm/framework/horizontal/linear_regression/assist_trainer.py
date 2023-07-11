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


from functools import partial
from .common import Common
from algorithm.core.horizontal.template.agg_type import register_agg_type_for_assist_trainer


class HorizontalLinearRegressionAssistTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        agg_type = list(self.common_config.aggregation["method"].keys())[0]
        register_agg_type_for_assist_trainer(self, 'torch', agg_type)
        self.register_hook(place="after_local_epoch", rank=1,
                           func=partial(self._save_model, False), desc="save model ")
        self.register_hook(place="after_local_epoch", rank=2,
                           func=partial(self.val_loop, "val"), desc="validation on valset")
        self.register_hook(place="after_global_epoch", rank=1,
                           func=partial(self._save_model, True), desc="save final model")
        
    def train_loop(self):
        pass
