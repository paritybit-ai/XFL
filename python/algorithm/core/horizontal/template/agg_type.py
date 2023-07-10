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


def register_agg_type_for_assist_trainer(trainer: object, framework: str, agg_type: str):
    if framework == 'torch':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.torch.fedavg.assist_trainer import FedAvgAssistTrainer
            FedAvgAssistTrainer(trainer).register()
        elif agg_type == "fedprox":
            from algorithm.core.horizontal.template.torch.fedprox.assist_trainer import FedProxAssistTrainer
            FedProxAssistTrainer(trainer).register()
        elif agg_type == "scaffold":
            from algorithm.core.horizontal.template.torch.scaffold.assist_trainer import SCAFFOLDAssistTrainer
            SCAFFOLDAssistTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg, fedprox, scaffold.")
    elif framework == 'tensorflow':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.tensorflow.fedavg.assist_trainer import FedAvgAssistTrainer
            FedAvgAssistTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg.")
    elif framework == 'jax':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.jax.fedavg.assist_trainer import FedAvgAssistTrainer
            FedAvgAssistTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg.")
    

def register_agg_type_for_label_trainer(trainer: object, framework: str, agg_type: str):
    if framework == 'torch':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.torch.fedavg.label_trainer import FedAvgLabelTrainer
            FedAvgLabelTrainer(trainer).register()
        elif agg_type == "fedprox":
            from algorithm.core.horizontal.template.torch.fedprox.label_trainer import FedProxLabelTrainer
            FedProxLabelTrainer(trainer).register()
        elif agg_type == "scaffold":
            from algorithm.core.horizontal.template.torch.scaffold.label_trainer import SCAFFOLDLabelTrainer
            SCAFFOLDLabelTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg, fedprox, scaffold.")
    elif framework == 'tensorflow':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.tensorflow.fedavg.label_trainer import FedAvgLabelTrainer
            FedAvgLabelTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg.")
    elif framework == 'jax':
        if agg_type == "fedavg":
            from algorithm.core.horizontal.template.jax.fedavg.label_trainer import FedAvgLabelTrainer
            FedAvgLabelTrainer(trainer).register()
        else:
            raise ValueError(f"Aggregation agg_type {agg_type} is not valid. Accepted agg_types are fedavg.")
