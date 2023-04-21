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
import pickle
import time
from typing import Any

from common.storage.redis.redis_conn import RedisConn
from common.utils.logger import logger
from service.fed_job import FedJob
from service.fed_node import FedNode
import commu_pb2
import scheduler_pb2_grpc
import trainer_pb2_grpc

MAX_BLOCK_SIZE = 1024 * 1024  # bytes


class Commu(object):
    """Implement peer to peer communication
    """
    fed_info = {}
    node = {}
    node_id = ""
    scheduler_id = ""
    trainer_ids = ""

    @classmethod
    def __init__(cls, fed_info: dict):
        # cls.* it to be deprecated
        cls.federal_info = fed_info
        cls.node = {}
        cls.node["scheduler"] = fed_info["scheduler"]
        cls.node.update(fed_info["trainer"])
        cls.node_id = fed_info["node_id"]
        cls.scheduler_id = "scheduler"
        cls.trainer_ids = list(fed_info["trainer"].keys())
        
    @classmethod
    def _get_channel(cls, remote_id: str):
        return FedNode.create_channel(remote_id)

    @classmethod
    def get_job_id(cls):
        return FedJob.job_id
        
    @classmethod
    def send(cls, key: str, value: Any, dst: str, use_pickle: bool = True) -> int:
        response = commu_pb2.PostResponse()
        channel = cls._get_channel(dst)
        if dst == "scheduler":
            stub = scheduler_pb2_grpc.SchedulerStub(channel)
        else:
            stub = trainer_pb2_grpc.TrainerStub(channel)
            
        request = commu_pb2.PostRequest()
        request.key = key
        
        if use_pickle:
            value = pickle.dumps(value)

        logger.debug(f"len of send msg: {len(value)}")
            
        def request_generator():
            n = math.ceil(1.0 * len(value) / MAX_BLOCK_SIZE)
            for i in range(n):
                request.value = value[i*MAX_BLOCK_SIZE: (i+1)*MAX_BLOCK_SIZE]
                yield request

        retry_num = 1
        sleep_sec = 1
        while True:
            try:
                response = stub.post(request_generator())
                break
            except Exception as ex:
                logger.warning(ex, exc_info=True)
                logger.warning(f"Send data retry {retry_num}...")
                retry_num += 1
                time.sleep(sleep_sec)
                if sleep_sec < 30:
                    sleep_sec *= 2

        return response.code
        
    @classmethod
    def recv(cls, key: str, use_pickle: bool = True, wait: bool = True, default_value: any = None) -> Any:
        if wait:
            data = RedisConn.cut(key)
        else:
            data = RedisConn.cut_if_exist(key)
            if data is None:
                return default_value
                
        if use_pickle:
            return pickle.loads(data)
        else:
            return data
