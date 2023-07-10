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
from pathlib import Path

import torch
from peft import (
    get_peft_model, PeftModel, PEFT_TYPE_TO_CONFIG_MAPPING
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments
)

from algorithm.core.data_io import QADataset
from service.fed_config import FedConfig
from common.utils.config_parser import CommonConfigParser
from common.utils.logger import logger
from common.checker.x_types import All
from common.utils.config_sync import ConfigSynchronizer


def alternateSVDLinear():
    import torch.nn.functional as F
    from peft.tuners.adalora import transpose
    
    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            result += (
                (
                    self.lora_dropout[self.active_adapter](x.float()) # to float()
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                    @ self.lora_B[self.active_adapter].T
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result
    
    from peft.tuners.adalora import SVDLinear
    
    SVDLinear.forward = forward


class Common:
    def __init__(self, train_conf: dict):
        if FedConfig.get_assist_trainer():
            sync_rule = {
                "train_info": {
                    "train_params": {
                        "aggregation": All(),
                        "encryption": All(),
                        "peft": All(),
                        "trainer": {
                            "learning_rate": All(),
                            "weight_decay": All(),
                            "adam_beta1": All(),
                            "adam_beta2": All(),
                            "adam_epsilon": All(),
                            "max_grad_norm": All(),
                            "max_steps": All(),
                            "num_train_epochs": All(),
                            "seed": All()
                        },
                        "dataset": All()
                    }
                }
            }     
            train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
            
        root_path = Path(__file__).parents[4]
        path = train_conf.get('input', {}).get("pretrained_model", {}).get("path")
        if path and not os.path.isabs(path):
            train_conf["input"]["pretrained_model"]['path'] = os.path.abspath(os.path.join(root_path, path))
        
        path = train_conf.get('input', {}).get("adapter_model", {}).get("path")
        if path and not os.path.isabs(path):
            train_conf["input"]["adapter_model"]['path'] = os.path.abspath(os.path.join(root_path, path))
            
        trainset_conf = train_conf.get('input', {}).get("trainset")
        if trainset_conf:
            path = trainset_conf[0].get("path")
            if path and not os.path.isabs(path):
                train_conf["input"]["trainset"][0]['path'] = os.path.abspath(os.path.join(root_path, path))
                
        self.common_config = CommonConfigParser(train_conf)
        
        # for adalora
        alternateSVDLinear()

        # CPU
        if self.common_config.train_params["trainer"].get("no_cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        pretrained_model_conf = self.common_config.input.get("pretrained_model", {})
        path = pretrained_model_conf.get("path")
        model_name_or_path = path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)  # Callback
        logger.info(self.tokenizer)
        
        self.load_from_pretrained = False
        for name in os.listdir(model_name_or_path):
            if 'pytorch_model' in name:
                self.load_from_pretrained = True
                break

        (peft_type, peft_config_dict), = self.common_config.train_params["peft"].items()

        if self.load_from_pretrained:
            if peft_type == "PREFIX_TUNING":
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                config.pre_seq_len = self.common_config.train_params["peft"][peft_type]["pre_seq_len"]
                config.prefix_projection = self.common_config.train_params["peft"][peft_type]["prefix_projection"]
                model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
            else:
                model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")
        else:
            if peft_type == "PREFIX_TUNING":
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                config.pre_seq_len = self.common_config.train_params["peft"][peft_type]["pre_seq_len"]
                config.prefix_projection = self.common_config.train_params["peft"][peft_type]["prefix_projection"]
                model = AutoModel.from_config(config, trust_remote_code=True)
            else: # if peft_type == "LORA":
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                model = AutoModel.from_config(config, trust_remote_code=True)
            logger.warning("No pretrained model founded, load from config")
        
        if self.common_config.train_params["trainer"].get("no_cuda"):
            model = model.float()
        else:
            model = model.half()
        
        if peft_type == "PREFIX_TUNING":
            self.model = model
        else:
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type](inference_mode=False, **peft_config_dict)
            # model = prepare_model_for_int8_training(model)
            self.model = get_peft_model(model, peft_config)
        
        if self.common_config.input.get("adapter_model", {}):
            adapter_path = self.common_config.input.get("adapter_model")["path"]
            if peft_type == "PREFIX_TUNING":
                # P-tuning v2
                prefix_state_dict = torch.load(os.path.join(adapter_path, "pytorch_model.bin"))
                new_prefix_state_dict = {}
                for k, v in prefix_state_dict.items():
                    if k.startswith("transformer.prefix_encoder."):
                        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
                self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            else:
                self.model = PeftModel.from_pretrained(self.model,
                                                       adapter_path,
                                                       adapter_name="default",
                                                       is_trainable=True)
            logger.info("Load adapter model.")
        
        if peft_type == "PREFIX_TUNING":
            if not self.common_config.train_params["trainer"].get("no_cuda"):
                self.model.transformer.prefix_encoder.float()
            
        logger.info(self.model)
        
        self.train_dataset, self.val_dataset = self._set_dataset()
        self.data_collator = self._set_data_collator()
        
        trainer_conf = self.common_config.train_params["trainer"]
        trainer_conf["max_steps"] = -1
        # trainer_conf["save_strategy"] = 'steps'
        trainer_conf["save_steps"] = self.common_config.train_params["aggregation"]["agg_steps"]
        trainer_conf["output_dir"] = self.common_config.output_dir
        self.training_args = TrainingArguments(**trainer_conf)
        self.trainer_conf = trainer_conf
        
        if peft_type == "PREFIX_TUNING":
            self.training_args.local_rank = -1
        
    def _set_data_collator(self):
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True
        )
        return data_collator
    
    def _set_dataset(self):
        dataset_conf = self.common_config.train_params["dataset"]
        train_dataset, val_dataset = None, None

        if self.common_config.input_trainset:
            file_name_or_path = os.path.join(
                self.common_config.input_trainset[0].get("path")
            )
            train_dataset = QADataset(
                file_name_or_path=file_name_or_path,
                tokenizer=self.tokenizer,
                max_src_length=dataset_conf["max_src_length"],
                max_dst_length=dataset_conf["max_dst_length"],
                prompt_pattern=dataset_conf["prompt_pattern"],
                key_query=dataset_conf.get("key_query", "input"),
                key_answer=dataset_conf.get("key_answer", "output")
            )
            
        if self.common_config.input_valset:
            file_name_or_path = os.path.join(
                self.common_config.input_valset[0].get("path")
            )
            
            val_dataset = QADataset(
                file_name_or_path=file_name_or_path,
                tokenizer=self.tokenizer,
                max_src_length=dataset_conf["max_src_length"],
                max_dst_length=dataset_conf["max_dst_length"],
                prompt_pattern=dataset_conf["prompt_pattern"],
                key_query=dataset_conf.get("key_query", "input"),
                key_answer=dataset_conf.get("key_answer", "output")
            )
        return train_dataset, val_dataset
    
