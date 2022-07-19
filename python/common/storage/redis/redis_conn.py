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
import time
from typing import Any

import redis

from common.utils.config import load_json_config
from service.fed_node import FedNode


class RedisConn(object):
    redis_config = {}
    retry_interval = None
    retry_duration = None
    rs = redis.StrictRedis()
    redis_host = ""

    @classmethod
    def init_redis(cls):
        config = load_json_config(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../", "common/storage/redis/data_pool_config.json")))

        cls.redis_config = config["redis"]
        cls.redis_host = FedNode.redis_host # or cls.redis_config.get("host")
        cls.redis_port = FedNode.redis_port # or cls.redis_config.get("port")
        cls.redis_config["host"] = cls.redis_host
        cls.redis_config["port"] = cls.redis_port
        cls.retry_interval = config.get("retry_interval")
        cls.retry_duration = config.get("retry_duration")

        pool = redis.ConnectionPool(host=cls.redis_config["host"], port=cls.redis_config["port"],
                                    db=0, decode_responses=False)
        cls.rs = redis.StrictRedis(connection_pool=pool)
        cls.init_job_id()

    @classmethod
    def init_job_id(cls):
        if cls.rs.get("XFL_JOB_ID") is None:
            cls.rs.set("XFL_JOB_ID", 0)

    @classmethod
    def put(cls, key: str, value: Any) -> int:
        status = cls.rs.set(key, value, ex=cls.redis_config["expire_seconds"])
        return status

    @classmethod
    def set(cls, key: str, value: Any, ex=-1) -> int:
        if ex > 0:
            return cls.rs.set(key, value, ex)
        else:
            return cls.rs.set(key, value)

    @classmethod
    def get(cls, key: str) -> Any:
        return cls.rs.get(key)

    @classmethod
    def incr(cls, key: str):
        return cls.rs.incr(key)

    @classmethod
    def cut(cls, key: str) -> Any:
        start = time.time()
        while True:
            if cls.rs.exists(key):
                res = cls.rs.get(key)
                cls.rs.delete(key)
                return res

            time.sleep(cls.retry_interval)
            if (time.time() - start) > cls.retry_duration:
                raise KeyError(f"Retry Timeout, Key {key} not found")

    @classmethod
    def cut_if_exist(cls, key: str) -> Any:
        time.sleep(1e-6)
        if cls.rs.exists(key):
            res = cls.rs.get(key)
            cls.rs.delete(key)
            return res
        else:
            return None
