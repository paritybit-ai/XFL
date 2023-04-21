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


from common.communication.gRPC.python import status_pb2
from common.storage.redis.redis_conn import RedisConn
from common.xoperator import get_operator


class FedJob(object):
    job_id = 0
    current_stage = 0
    total_stage_num = 0
    global_epoch = None
    algo_info = None

    status = status_pb2.IDLE
    progress = []
    max_progress = 100

    @classmethod
    def init_fedjob(cls):
        cls.job_id = int(RedisConn.get("XFL_JOB_ID"))

    @classmethod
    def init_progress(cls, total_stage_num):
        cls.total_stage_num = total_stage_num
        cls.progress = [0] * total_stage_num

    @classmethod
    def get_model(cls, role: str, stage_config: dict) -> object:
        """Get model handler

        Args:
            role (str): The role this node played in the federation. Supported roles are "assist_trainer", "label_trainer" and "trainer".
            stage_config (dict):

        Returns:
            Model handler. 
        """
        model_name = stage_config["model_info"]["name"]
        model = get_operator(name=model_name, role=role)
        return model(stage_config)
