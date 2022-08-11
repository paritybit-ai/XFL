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


from service.fed_job import FedJob
from service.fed_node import FedNode


class TrainConfigParser(object):
    def __init__(self, config: dict) -> None:
        self.train_conf = config
        self.identity = config.get("identity")
        self.fed_config = config.get("fed_info")
        self.model_info = config.get("model_info")
        self.inference = config.get("inference", False)
        self.train_info = config.get("train_info")
        self.extra_info = config.get("extra_info")
        self.computing_engine = config.get("computing_engine", "local")
        self.device = self.train_info.get("device", "cpu")

        if self.train_info:
            self.train_params = self.train_info.get("params")
            self.interaction_params = self.train_info.get("interaction_params", {})
        else:
            self.train_params = None
            self.interaction_params = None

        self.input = config.get("input")
        if self.input:
            for i in ["dataset", "trainset", "valset", "testset"]:
                for j in range(len(self.input.get(i, []))):
                    if "path" in self.input[i][j]:
                        self.input[i][j]["path"] = self.input[i][j]["path"] \
                            .replace("[JOB_ID]", str(FedJob.job_id)) \
                            .replace("[NODE_ID]", str(FedNode.node_id))
            self.input_trainset = self.input.get("trainset", [])
            self.input_valset = self.input.get("valset", [])
            self.input_testset = self.input.get("testset", [])
        else:
            self.input_trainset = []
            self.input_valset = []
            self.input_testset = []

        self.output = config.get("output")
        if self.output:
            for i in ["dataset", "trainset", "valset", "testset", "model", "metrics", "evaluation"]:
                if self.output.get(i) is not None and "path" in self.output.get(i):
                    self.output.get(i)["path"] = self.output.get(i)["path"] \
                        .replace("[JOB_ID]", str(FedJob.job_id)) \
                        .replace("[NODE_ID]", str(FedNode.node_id))
