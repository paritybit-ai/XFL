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


from algorithm.core.horizontal.template.torch.fedtype import _get_assist_trainer
from functools import partial
from .common import Common


class HorizontalResnetAssistTrainer(Common, _get_assist_trainer()):
    def __init__(self, train_conf: dict):
        _get_assist_trainer().__init__(self, train_conf)
        
        self.register_hook(place="after_local_epoch", rank=2,
                           func=partial(self.val_loop, "val"), desc="validation on valset")
        self.register_hook(place="after_local_epoch", rank=3,
                           func=self._save_model, desc="save model")
        
    def train_loop(self):
        pass
