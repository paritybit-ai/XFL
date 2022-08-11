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


import multiprocessing
import time
import traceback
from concurrent import futures
from multiprocessing import Process

import grpc

from common.communication.gRPC.python import (control_pb2, scheduler_pb2_grpc,
                                              status_pb2, trainer_pb2_grpc)
from common.communication.gRPC.python.commu import Commu
from common.storage.redis.redis_conn import RedisConn
from common.utils.grpc_channel_options import insecure_options
from common.utils.logger import logger, remove_log_handler
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode
from service.trainer import TrainerService

multiprocessing.set_start_method('fork')


def start_trainer_service(status):
    FedJob.status = status
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=insecure_options)
    trainer_pb2_grpc.add_TrainerServicer_to_server(TrainerService(), server)
    FedNode.add_server(server)
    server.start()
    logger.info("Trainer Service Start...")
    logger.info(f"[::]:{FedNode.listening_port}")
    server.wait_for_termination()


def start_server():
    status = multiprocessing.Value("i", status_pb2.IDLE)
    p = Process(target=start_trainer_service, args=(status,))
    p.start()

    while True:
        time.sleep(1)
        try:
            if status.value == status_pb2.START_TRAIN:
                FedJob.process = Process(target=train, args=(status,))
                FedJob.process.start()
                status.value = status_pb2.TRAINING

            elif status.value == status_pb2.STOP_TRAIN:
                if FedJob.process is not None:
                    FedJob.process.terminate()
                    logger.info("Model training is stopped.")
                    FedJob.process = None
                status.value = status_pb2.FAILED

            remove_log_handler(FedConfig.job_log_handler)
            remove_log_handler(FedConfig.job_stage_log_handler)
        except Exception:
            logger.error(traceback.format_exc())
            remove_log_handler(FedConfig.job_log_handler)
            remove_log_handler(FedConfig.job_stage_log_handler)
            FedJob.status.value = status_pb2.FAILED


def train(status):
    try:
        FedConfig.get_config()
        identity = FedConfig.stage_config["identity"]
        inference = FedConfig.stage_config.get("inference", False)
        logger.info(f"{identity} Start Training...")
        model = FedJob.get_model(identity, FedConfig.stage_config)
        if inference:
            model.predict()
        else:
            model.fit()
        status.value = status_pb2.SUCCESSFUL
        logger.info("Train Model Successful.")
    except Exception as ex:
        logger.error(ex, exc_info=True)
        logger.warning("Train Model Failed.")
        job_control(control_pb2.STOP)


def job_control(control):
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = control_pb2.ControlRequest()
    request.control = control
    response = stub.control(request)
    logger.info(response)


def main(identity, debug_node_id, config_path=''):
    FedNode.init_fednode(identity=identity, debug_node_id=debug_node_id, conf_dir=config_path)
    RedisConn.init_redis()
    Commu(FedNode.config)
    start_server()
