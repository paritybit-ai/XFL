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


from algorithm.core.horizontal.template.jax.fedtype import _get_label_trainer
from common.utils.logger import logger
from .common import Common
from jax import jit, value_and_grad


class HorizontalVggJaxLabelTrainer(Common, _get_label_trainer()):
    def __init__(self, train_conf: dict):
        _get_label_trainer().__init__(self, train_conf)
        self._set_jit_train_step()
        self._set_jit_val_step()

    def _set_jit_train_step(self):
        def train_step(batch, state):
            loss_fn = lambda params: self.calculate_loss(params, state.batch_stats, batch, train=True)
            ret, grads = value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, _, new_model_state = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return loss, state
        self.jit_train_step = jit(train_step)
        
    def train_loop(self):
        train_loss = 0
        
        for batch_id, batch in enumerate(self.train_dataloader):
            loss, self.state = self.jit_train_step(batch, self.state)
            train_loss += loss.item()
        train_loss /= len(self.train_dataloader)
        self.context["train_loss"] = train_loss
        logger.info(f"Train loss: {train_loss}")
