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


from service.fed_config import FedConfig

def _get_assist_trainer():
    aggregation_config = FedConfig.stage_config["train_info"]["params"]["aggregation_config"]
    type = aggregation_config.get("type")
    if type == "fedprox":
        from python.algorithm.core.horizontal.template.torch.fedprox.assist_trainer import FedProxAssistTrainer
        return FedProxAssistTrainer
        
    from python.algorithm.core.horizontal.template.torch.fedavg.assist_trainer import FedAvgAssistTrainer
    return FedAvgAssistTrainer

def _get_label_trainer():
    aggregation_config = FedConfig.stage_config["train_info"]["params"]["aggregation_config"]
    type = aggregation_config.get("type")
    if type == "fedprox":
        from python.algorithm.core.horizontal.template.torch.fedprox.label_trainer import FedProxLabelTrainer
        return FedProxLabelTrainer

    from python.algorithm.core.horizontal.template.torch.fedavg.label_trainer import FedAvgLabelTrainer
    return FedAvgLabelTrainer