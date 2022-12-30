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


import datetime
import time
import traceback
from concurrent import futures

import grpc

from common.communication.gRPC.python import (control_pb2, scheduler_pb2_grpc,
                                              status_pb2)
from common.communication.gRPC.python.commu import Commu
from common.storage.redis.redis_conn import RedisConn
from common.utils.grpc_channel_options import insecure_options
from common.utils.logger import logger, remove_log_handler
from service.fed_config import FedConfig
from service.fed_control import get_trainer_status, trainer_control
from service.fed_job import FedJob
from service.fed_node import FedNode
from service.scheduler import SchedulerService


def start_server(config_path):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=insecure_options)
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(SchedulerService(), server)
    FedNode.add_server(server)
    server.start()
    logger.info("Scheduler Service Start...")
    logger.info(f"[::]:{FedNode.listening_port}")

    while True:
        time.sleep(1)
        try:
            if FedJob.status == status_pb2.TRAINING:
                start_time = datetime.datetime.now()
                FedConfig.load_config(config_path)
                for stage in range(len(FedConfig.trainer_config)):
                    logger.info(f"Stage {stage} Start...")
                    FedJob.current_stage = stage
                    trainer_control(control_pb2.START)

                    trainer_status = {}
                    while True:
                        time.sleep(1)
                        resp = get_trainer_status()
                        for i in resp.keys():
                            if resp[i].code == status_pb2.FAILED:
                                FedJob.status = status_pb2.FAILED
                                logger.warning(f"Stage {stage} Failed.")
                                break
                            elif resp[i].code == status_pb2.SUCCESSFUL:
                                trainer_status[i] = resp[i].code
                        if FedJob.status == status_pb2.FAILED:
                            break
                        elif len(trainer_status) == len(FedNode.trainers):
                            logger.info(f"Stage {stage} Successful.")
                            break
                    if FedJob.status == status_pb2.FAILED:
                        break

                if FedJob.status == status_pb2.TRAINING:
                    logger.info("All Stage Successful.")
                    logger.info(f"JOB_ID: {FedJob.job_id} Successful.")
                    RedisConn.set("XFL_JOB_STATUS_"+str(FedJob.job_id), status_pb2.SUCCESSFUL)
                    FedJob.status = status_pb2.SUCCESSFUL
                else:
                    logger.warning(f"JOB_ID: {FedJob.job_id} Failed.")
                    trainer_control(control_pb2.STOP)
                    RedisConn.set("XFL_JOB_STATUS_"+str(FedJob.job_id), status_pb2.FAILED)

                end_time = datetime.datetime.now()
                cost_time = (end_time - start_time).seconds
                logger.info(f"Cost time: {cost_time} seconds.")

                remove_log_handler(FedConfig.job_log_handler)
        except Exception:
            logger.error(traceback.format_exc())
            remove_log_handler(FedConfig.job_log_handler)
            FedJob.status = status_pb2.FAILED

    server.wait_for_termination()


def main(config_path):
    FedNode.init_fednode(conf_dir=config_path)
    RedisConn.init_redis()
    FedJob.init_fedjob()
    FedConfig.load_algorithm_list()
    Commu(FedNode.config)
    start_server(config_path)
