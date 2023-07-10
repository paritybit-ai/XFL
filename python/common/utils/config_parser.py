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


# from service.fed_job import FedJob
# from service.fed_node import FedNode


def replace_variable(input, stage_id: int, job_id: str, node_id: str):
    if isinstance(input, dict):
        return {k: replace_variable(v, stage_id, job_id, node_id) for k, v in input.items()}
    elif isinstance(input, list):
        return [replace_variable(v, stage_id, job_id, node_id) for v in input]
    elif isinstance(input, str):
        input = input.replace("[STAGE_ID]", str(stage_id)).replace("[JOB_ID]", str(job_id)).replace("[NODE_ID]", str(node_id))
        if "STAGE_ID" in input:
            start = -1
            for idx, c in enumerate(input):
                if c == '[':
                    start = idx
                elif c == ']':
                    end = idx
                    if start != -1:
                        s = input[start+1: end]
                        s = s.replace("STAGE_ID", str(stage_id))
                        nums = s.split('-')
                        if len(nums) == 2:
                            stage_id = int(nums[0]) - int(nums[1])
                            input = input.replace(input[start: end+1], str(stage_id))
                    start = -1
                
        return input
    else:
        return input


class TrainConfigParser(object):
    def __init__(self, config: dict) -> None:
        self.train_conf = config
        self.inference = config.get("inference", False)
        
        self.identity = config.get("identity")
        self.fed_config = config.get("fed_info")
        self.model_info = config.get("model_info")
        
        self.train_info = config.get("train_info", {})
        self.extra_info = config.get("extra_info")
        self.computing_engine = config.get("computing_engine", "local")
        self.device = self.train_info.get("device", "cpu")

        self.train_params = self.train_info.get("params") or self.train_info.get("train_params")
        self.interaction_params = self.train_info.get("interaction_params", {})
        self.save_frequency = self.interaction_params.get("save_frequency", -1)
        self.write_training_prediction = \
            self.interaction_params.get("write_training_prediction", False)
        self.write_validation_prediction = \
            self.interaction_params.get("write_validation_prediction", False)
        
        self.input = config.get("input", {})
        self.input_trainset = self.input.get("trainset", [])
        self.input_valset = self.input.get("valset", [])
        self.input_testset = self.input.get("testset", [])

        self.output = config.get("output")


class CommonConfigParser:
    # Parse the original config.json to extract common config fields
    def __init__(self, config: dict) -> None:
        self.config = config
        
        self.identity = config.get("identity")
        self.model_info = config.get("model_info", {})
        self.model_conf = self.model_info.get("config", {})

        self.input = config.get("input", {})
        self.input_trainset = self.input.get("trainset", [])
        self.input_valset = self.input.get("valset", [])
        self.input_testset = self.input.get("testset", [])
        self.pretrain_model = self.input.get("pretrain_model", {})
        self.pretrain_model_path = self.pretrain_model.get("path", "")
        self.pretrain_model_name = self.pretrain_model.get("name", "")

        self.output = config.get("output", {})
        self.output_dir = self.output.get("path", "")
        self.output_model_name = self.output.get("model", {}).get("name", "")
        self.output_onnx_model_name = self.output.get("onnx_model", {}).get("name", "")
        
        self.train_info = config.get("train_info", {})
        self.device = self.train_info.get("device", "cpu")

        self.interaction_params = self.train_info.get("interaction_params", {})
        self.save_frequency = self.interaction_params.get("save_frequency", -1)
        self.echo_training_metrics = self.interaction_params.get("echo_training_metrics", False)
        self.write_training_prediction = \
            self.interaction_params.get("write_training_prediction", False)
        self.write_validation_prediction = \
            self.interaction_params.get("write_validation_prediction", False)
        
        self.train_params = self.train_info.get("train_params", {})
        self.aggregation = self.train_params.get("aggregation", {})
        self.encryption = self.train_params.get("encryption", {"plain": {}})

        self.optimizer = self.train_params.get("optimizer", {})
        self.lr_scheduler = self.train_params.get("lr_scheduler", {})
        self.lossfunc = self.train_params.get("lossfunc", {})
        self.metric = self.train_params.get("metric", {})
        self.early_stopping = self.train_params.get("early_stopping", {})

        self.random_seed = self.train_params.get("random_seed", None)