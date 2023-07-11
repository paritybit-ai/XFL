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


import math
import inspect
from pathlib import Path
from typing import Union
from copy import deepcopy

import torch
import torch.nn as nn
import transformers

from accelerate import (
    dispatch_model, infer_auto_device_map
)
from accelerate.utils import get_balanced_memory
from accelerate.hooks import (
    AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
)
from transformers import (
    TrainerCallback, TrainingArguments, TrainerState, TrainerControl
)
from peft import PeftModel
from peft.utils import (
    get_peft_model_state_dict, set_peft_model_state_dict, PromptLearningConfig
)

from algorithm.core.horizontal.aggregation.api import (
    get_aggregation_root_inst, get_aggregation_leaf_inst
)
from common.utils.logger import logger
from service.fed_config import FedConfig


# def is_nan_exists(state_dict):
#     flag = False
#     for k, v in state_dict.items():
#         if torch.isnan(v).any():
#             flag = True
#             logger.warning(f"Parameter {k} contains nan")
#             break
#     return flag


class AssistTrainerCallback(TrainerCallback):
    def __init__(self,
                 agg_steps: int,
                 sec_conf: dict,
                 root_id: str,
                 leaf_ids: list[str],
                 init_params: bool = False,
                 peft_type: str = "LORA"):
        super().__init__()
        self.agg_steps = agg_steps
        self.agg_steps_list = []
        assert 0 < agg_steps <= 1
        self.agg_inst = get_aggregation_root_inst(sec_conf, root_id, leaf_ids)
        self.init_params = init_params
        self.peft_type = peft_type
        # self.latest_adapters_weights = None

    def on_train_begin(self,
                       args: TrainingArguments,
                       state: TrainerState,
                       control: TrainerControl,
                       model: Union[transformers.PreTrainedModel, torch.nn.Module],
                       **kwargs):
        if self.init_params:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    torch.nn.init.uniform_(m.weight.data)
        args.logging_steps = state.max_steps + 1
        args.save_steps = state.max_steps + 1
        
        adapters_weights = get_adapter_state_dict(model, self.peft_type)
        self.agg_inst.broadcast(adapters_weights)
        # self.latest_adapters_weights = adapters_weights

    # def on_step_begin(self,
    #                   args: TrainingArguments,
    #                   state: TrainerState,
    #                   control: TrainerControl,
    #                   model: Union[transformers.PreTrainedModel, torch.nn.Module],
    #                   **kwargs):
    #     for k, v in model.state_dict().items():
    #         if torch.isnan(v).any():
    #             # set nan to 0
    #             v[v != v] = 0
    #             logger.warning(f"Parameter {k} contains nan, replace nan to 0")
    #     control.should_log = False

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    model: Union[transformers.PreTrainedModel, torch.nn.Module],
                    tokenizer,
                    **kwargs):
        # Trainer saves model after check on_step_end
        if not self.agg_steps_list:
            i = self.agg_steps
            while i < 1:
                self.agg_steps_list.append(round(i, 4))
                i += self.agg_steps
            self.agg_steps_list.append(1)
            self.steps_list = [math.ceil(i * state.max_steps)
                               for i in self.agg_steps_list]
            assert len(self.steps_list) == len(set(self.steps_list))
            logger.info(f"Aggergate model by steps: {self.agg_steps_list}")

        if state.global_step in self.steps_list:
            idx = self.steps_list.index(state.global_step)
            if idx == 0:
                factor = 1
            else:
                factor = 1

            adapters_weights = get_adapter_state_dict(model, self.peft_type)
            logger.info(
                f"gather and aggregating..., global_step={state.global_step}")
            new_adapters_weights = self.agg_inst.aggregate(
                parameters=adapters_weights, parameters_weight=factor)
            
            set_adapter_state_dict(model, self.peft_type, new_adapters_weights)
            logger.info(f"broadcasting..., global_step={state.global_step}")
            self.agg_inst.broadcast(new_adapters_weights)
            if args.output_dir and args.save_strategy != 'no':
                if self.peft_type != "PREFIX_TUNING":
                    model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}")
                else:
                    model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}",
                                          state_dict=get_adapter_state_dict(model, self.peft_type))
            control.should_log = True


class LabelTrainerCallback(TrainerCallback):
    def __init__(self,
                 agg_steps: Union[float, int],
                 sec_conf: dict,
                 root_id: str,
                 leaf_ids: list[str],
                 init_params: bool = False,
                 peft_type: str = "LORA"):
        super().__init__()
        self.agg_steps = agg_steps
        self.agg_steps_list = []
        assert 0 < agg_steps <= 1
        self.is_standalone = False if FedConfig.get_assist_trainer() else True
        if not self.is_standalone:
            self.agg_inst = get_aggregation_leaf_inst(sec_conf, root_id, leaf_ids)
        self.init_params = init_params
        self.peft_type = peft_type

    def on_train_begin(self,
                       args: TrainingArguments,
                       state: TrainerState,
                       control: TrainerControl,
                       model: Union[transformers.PreTrainedModel, torch.nn.Module],
                       train_dataloader,
                       **kwargs):
        if self.init_params:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    torch.nn.init.uniform_(m.weight.data)

        args.logging_steps = state.max_steps + 1
        args.save_steps = state.max_steps + 1
        
        if not self.is_standalone:
            new_adapters_weights = self.agg_inst.download()
            set_adapter_state_dict(model, self.peft_type, new_adapters_weights)

    # def on_step_begin(self,
    #                   args: TrainingArguments,
    #                   state: TrainerState,
    #                   control: TrainerControl,
    #                   model: Union[transformers.PreTrainedModel, torch.nn.Module],
    #                   **kwargs):
    #     for k, v in model.state_dict().items():
    #         if torch.isnan(v).any():
    #             # set nan to 0
    #             v[v != v] = 0
    #             logger.warning(f"Parameter {k} contains nan, replace nan to 0")
    #     control.should_log = False

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    model: Union[transformers.PreTrainedModel, torch.nn.Module],
                    **kwargs):
        # if is_nan_exists(model.state_dict()):
        #     logger.warning(f"Nan exists!")
            
        # Trainer saves model after check on_step_end
        if not self.agg_steps_list:
            i = self.agg_steps
            while i < 1:
                self.agg_steps_list.append(round(i, 4))
                i += self.agg_steps
            self.agg_steps_list.append(1)
            self.steps_list = [math.ceil(i * state.max_steps)
                               for i in self.agg_steps_list]
            if len(self.steps_list) != len(set(self.steps_list)):
                raise ValueError(f"agg_steps is too small, try a larger one.")
            logger.info(f"Aggergate model by steps: {self.agg_steps_list}")

        if state.global_step in self.steps_list:
            idx = self.steps_list.index(state.global_step)
            if not self.is_standalone:
                if idx == 0:
                    factor = 1
                    # factor = self.agg_steps_list[0] * \
                    #     args.gradient_accumulation_steps * args.train_batch_size
                else:
                    factor = 1
                    # factor = (self.agg_steps_list[idx] - self.agg_steps_list[idx-1]) * \
                    #     args.gradient_accumulation_steps * args.train_batch_size
                adapters_weights = get_adapter_state_dict(model, self.peft_type)
                logger.info(f"uploading..., global_step={state.global_step}")
                self.agg_inst.upload(adapters_weights, factor)
                logger.info(f"downloading..., global_step={state.global_step}")
                new_adapters_weights = self.agg_inst.download()
                set_adapter_state_dict(model, self.peft_type, new_adapters_weights)

            if args.output_dir and args.save_strategy != 'no':
                if self.peft_type != "PREFIX_TUNING":
                    model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}")
                else:
                    model.save_pretrained(save_directory=Path(args.output_dir) / f"checkpoint-{str(self.agg_steps_list[idx])}",
                                          state_dict=get_adapter_state_dict(model, self.peft_type))

            control.should_log = True


def get_adapter_state_dict(model: PeftModel, peft_type: str, **kwargs):
    if peft_type == "PREFIX_TUNING":
        state_dict = model.state_dict()
        adapters_weights = {}
        for k, v in model.named_parameters():
            if v.requires_grad:
                adapters_weights[k] = deepcopy(state_dict[k]).to('cpu')
    else: # if peft_type == 'LORA':
        adapter_name = model.active_adapter
        adapters_weights = get_peft_model_state_dict(
            model, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
        )

        for k, v in adapters_weights.items():
            adapters_weights[k] = deepcopy(v).to('cpu')
    return adapters_weights


def set_adapter_state_dict(model: PeftModel, peft_type: str, adapters_weights: dict, **kwargs):
    
    if peft_type == "PREFIX_TUNING":
        state_dict = model.state_dict()
        state_dict.update(adapters_weights)
        model.load_state_dict(state_dict)
    else:
        adapter_name = model.active_adapter
        # load the weights into the model
        set_peft_model_state_dict(model, adapters_weights, adapter_name=adapter_name)
        if (
            (getattr(model, "hf_device_map", None) is not None)
            and (len(set(model.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(model.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = model._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(model.peft_config[adapter_name], PromptLearningConfig):
                remove_hook_from_submodules(model.prompt_encoder)
            add_hook_to_module(model.get_base_model(), hook)

    # Set model in evaluation mode to deactivate Dropout modules by default
    # model.eval()
