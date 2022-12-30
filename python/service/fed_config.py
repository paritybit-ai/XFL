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


import json
import os

from common.communication.gRPC.python import scheduler_pb2, scheduler_pb2_grpc
from common.utils.config import load_json_config
from common.utils.logger import (add_job_log_handler,
                                 add_job_stage_log_handler, logger)
from common.xoperator import get_operator
from service.fed_job import FedJob
from service.fed_node import FedNode


class FedConfig(object):
    trainer_config = {}
    stage_config = {}
    algorithm_list = []
    default_config_map = {}
    job_log_handler = None
    job_stage_log_handler = None
    
    @classmethod
    @property
    def job_id(cls):
        return FedJob.job_id
    
    @classmethod
    @property
    def node_id(cls):
        return FedNode.node_id
    
    @classmethod
    @property
    def redis_host(cls):
        return FedNode.redis_host
    
    @classmethod
    @property
    def redis_port(cls):
        return FedNode.redis_port

    @classmethod
    def get_label_trainer(cls):
        res = cls.stage_config.get("fed_info", {}).get("label_trainer", [])
        return res
        
    @classmethod
    def get_assist_trainer(cls):
        res = cls.stage_config.get("fed_info", {}).get("assist_trainer", [])
        if len(res) > 0:
            return res[0]
        else:
            return None
        
    @classmethod
    def get_trainer(cls):
        res = cls.stage_config.get("fed_info", {}).get("trainer", [])
        return res
    
    @classmethod
    def load_config(cls, config_path):
        cls.job_log_handler = add_job_log_handler(FedJob.job_id)
        logger.info("Loading Config...")
        cls.trainer_config = cls.load_trainer_config(config_path)
        logger.info("Load Config Completed.")

    # @classmethod
    # def load_trainer_config(cls, config_path):
    #     trainer_config = {}
    #     for node_id in FedNode.trainers.keys():
    #         info = load_json_config(f"{config_path}/trainer_config_{node_id}.json")
    #         for idx in range(len(info)):
    #             if idx not in trainer_config.keys():
    #                 trainer_config[idx] = {}
    #             trainer_config[idx][node_id] = info[idx]
                    
    #     for stage_id in trainer_config:
    #         fed_info = {
    #             "label_trainer": [],
    #             "trainer": [],
    #             "assist_trainer": []
    #         }
    #         for node_id in trainer_config[stage_id]:
    #             # identity = trainer_config[stage_id][node_id]["identity"]
    #             identity = trainer_config[stage_id][node_id].get("identity")
    #             if identity:
    #                 fed_info[identity].append(node_id)
    #             trainer_config[stage_id][node_id]["fed_info"] = fed_info

    #     return trainer_config
    
    @classmethod
    def load_trainer_config(cls, config_path):
        trainer_config = {}
        unconfiged_node_ids = []
        op_names = {}
        for node_id in FedNode.trainers.keys():
            f_path = f"{config_path}/trainer_config_{node_id}.json"
            if not os.path.exists(f_path):
                unconfiged_node_ids.append(node_id)
                continue
            
            info = load_json_config(f_path)
            for stage_id in range(len(info)):
                if stage_id not in trainer_config.keys():
                    trainer_config[stage_id] = {}
                    op_names[stage_id] = []
                trainer_config[stage_id][node_id] = info[stage_id]
                
                op_name = info[stage_id].get("model_info", {}).get("name")
                if op_name:
                    op_names[stage_id].append(op_name)
                
        if len(unconfiged_node_ids) > 1:
            logger.warning(f"{len(unconfiged_node_ids)} nodes-{unconfiged_node_ids} are not configed.")
            
        if len(unconfiged_node_ids) == 1:
            assist_trainer_id = unconfiged_node_ids[0]
        
            for stage_id in op_names:
                if len(set(op_names[stage_id])) != 1:
                    logger.warning(f"Operator names {op_names[stage_id]} not the same in stage {stage_id}.")
                    continue
                
                op_name = op_names[stage_id][0]
                try:
                    operator = get_operator(op_name, "assist_trainer")
                except Exception:
                    operator = None

                if operator is not None:
                    assist_trainer_config = {
                        "identity": "assist_trainer",
                        "model_info": {
                            "name": op_name
                        },
                    }
                else:
                    assist_trainer_config = {}
                
                trainer_config[stage_id][assist_trainer_id] = assist_trainer_config
                    
        for stage_id in trainer_config:
            fed_info = {
                "label_trainer": [],
                "trainer": [],
                "assist_trainer": []
            }
            for node_id in trainer_config[stage_id]:
                identity = trainer_config[stage_id][node_id].get("identity")
                if identity:
                    fed_info[identity].append(node_id)
                trainer_config[stage_id][node_id]["fed_info"] = fed_info

        return trainer_config

    @classmethod
    def get_config(cls):
        request = scheduler_pb2.GetConfigRequest()
        request.nodeId = FedNode.node_id
        channel = FedNode.create_channel("scheduler")
        stub = scheduler_pb2_grpc.SchedulerStub(channel)
        response = stub.getConfig(request)
        cls.stage_config = json.loads(response.config)
        FedJob.job_id = response.jobId
        cls.job_log_handler = add_job_log_handler(FedJob.job_id)
        cls.job_stage_log_handler = add_job_stage_log_handler(
            FedJob.job_id, FedConfig.stage_config.get("model_info", {}).get("name", ""))
        if "global_epoch" in cls.stage_config.get("train_info", {}).get("train_params", {}):
            FedJob.global_epoch = cls.stage_config["train_info"].get("train_params", {}).get("global_epoch")

        logger.info("stage_config: " + str(cls.stage_config))

        return response

    @classmethod
    def load_algorithm_list(cls):
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../algorithm/config/'))
        
        algo_info = {}
        
        for algorithm_name in os.listdir(config_dir):
            algorithm_conf_dir = os.path.join(config_dir, algorithm_name)
            if not os.path.isdir(algorithm_conf_dir):
                continue
            algo_info[algorithm_name] = []
                
            for party_conf_file in os.listdir(algorithm_conf_dir):
                file_name = party_conf_file.split(".")[0]
                if file_name != "__init__":
                    algo_info[algorithm_name].append(file_name)
        
        cls.algorithm_list = list(algo_info.keys())

        for k in cls.algorithm_list:
            dc = {}
            for v in algo_info[k]:
                conf = load_json_config(os.path.abspath(
                    os.path.join(os.path.dirname(__file__), f'../algorithm/config/{k}/{v}.json')))
                dc[v] = conf
            cls.default_config_map[k] = dc
