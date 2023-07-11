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


from pathlib import Path
from transformers import Trainer
from algorithm.core.horizontal.aggregation.api import get_aggregation_root_inst
from service.fed_config import FedConfig
from common.utils.logger import logger
from algorithm.framework.horizontal.chatglm.common import Common
from algorithm.framework.horizontal.chatglm.callback import (
    AssistTrainerCallback, get_adapter_state_dict, set_adapter_state_dict
)


class HorizontalChatglmAssistTrainer(Common):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)

    def fit(self):
        agg_steps = self.common_config.aggregation.get("agg_steps") or self.common_config.train_params["trainer"]["save_steps"]
        sec_conf = self.common_config.train_params["encryption"]
        (peft_type, peft_config_dict), = self.common_config.train_params["peft"].items()
        
        if len(self.common_config.input_trainset) != 0:
            my_callback = AssistTrainerCallback(agg_steps,
                                                sec_conf,
                                                root_id=FedConfig.get_assist_trainer(),
                                                leaf_ids=FedConfig.get_label_trainer(),
                                                init_params=not self.load_from_pretrained,
                                                peft_type=peft_type)
            
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                callbacks=[my_callback],
            )
            
            trainer.train()
        else:
            agg_inst = get_aggregation_root_inst(sec_conf,
                                                 root_id=FedConfig.get_assist_trainer(),
                                                 leaf_ids=FedConfig.get_label_trainer())
            
            self.agg_steps_list = []
            i = agg_steps
            while i < 1:
                self.agg_steps_list.append(round(i, 4))
                i += agg_steps
            self.agg_steps_list.append(1)
            
            adapters_weights = get_adapter_state_dict(self.model, peft_type)
            agg_inst.broadcast(adapters_weights)
            
            for i in range(len(self.agg_steps_list)):
                logger.info(f"gather and agg, global_step={self.agg_steps_list[i]}")
                new_adapters_weights = agg_inst.aggregate()
                logger.info(f"broadcast, global_step={self.agg_steps_list[i]}")
                agg_inst.broadcast(new_adapters_weights)

                if self.training_args.output_dir and self.training_args.save_strategy != 'no':
                    save_dir = Path(self.common_config.output["path"]) / f"checkpoint-{str(self.agg_steps_list[i])}"
                    if peft_type == "PREFIX_TUNING":
                        self.model.save_pretrained(save_directory=save_dir,
                                                   state_dict=new_adapters_weights)
                    else:
                        set_adapter_state_dict(self.model, peft_type, new_adapters_weights)
                        self.model.save_pretrained(save_directory=save_dir)

                
