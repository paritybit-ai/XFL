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


from common.communication.gRPC.python import (
    control_pb2, status_pb2, trainer_pb2_grpc, scheduler_pb2, scheduler_pb2_grpc
)
from common.utils.logger import logger
from service.fed_node import FedNode


class ProgressCalculator:
    def __init__(self, *args):
        self.param_num = len(args)
        self.iter_list = []
        self.max_list = args
        self.tick_list = [100]
        for max_item in self.max_list:
            last_tick = self.tick_list[-1]
            assert max_item > 0
            self.tick_list.append(last_tick / max_item)
            
    def cal_progress(self):
        '''
            the iter_item always begin from 1.
        '''
        progress = self.tick_list[-1]
        for iter_item, tick_item in zip(self.iter_list, self.tick_list[1:]):
            progress += (iter_item - 1) * tick_item
        
        _send_progress(int(progress))
    
    def cal_custom_progress(self, *args):
        if len(args) != self.param_num:
            raise ValueError("The number of args is not equal to the number of max values.")
        self.iter_list = args
        self.cal_progress()
    
    def cal_horizontal_progress(self, context: dict):
        self.iter_list = [context["g_epoch"]]
        if len(self.iter_list) != self.param_num:
            raise ValueError("The number of args is not equal to the number of max values.")
        self.cal_progress()
    
    @staticmethod
    def finish_progress(context: dict=None):
        _send_progress(100)


def _send_progress(progress):
    progress = progress if progress <= 100 else 100
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = scheduler_pb2.RecProgressRequest()
    # request.stageId = FedJob.current_stage
    request.progress = progress
    stub.recProgress(request)
    return


def trainer_control(control):
    response = control_pb2.ControlResponse()
    for node_id in FedNode.trainers.keys():
        channel = FedNode.create_channel(node_id)
        stub = trainer_pb2_grpc.TrainerStub(channel)
        request = control_pb2.ControlRequest()
        request.control = control
        try:
            resp = stub.control(request)
            if resp.code == 0:
                response.message += f"{control_pb2.Operation.Name(control)} Trainer: {node_id} Successful.\n"
            else:
                response.code = 1
                response.message += f"{control_pb2.Operation.Name(control)} Trainer: {node_id} Failed.\n"
        except Exception as ex:
            logger.error(ex, exc_info=True)
            logger.error(f"{control_pb2.Operation.Name(control)} Trainer: {node_id} Failed.")
            response.code = 1
            response.message += f"{control_pb2.Operation.Name(control)} Trainer: {node_id} Failed.\n"
    return response


def get_trainer_status():
    response = status_pb2.StatusResponse()
    for node_id in FedNode.trainers.keys():
        channel = FedNode.create_channel(node_id)
        stub = trainer_pb2_grpc.TrainerStub(channel)
        request = status_pb2.StatusRequest()
        node_status = status_pb2.Status()
        try:
            resp = stub.status(request)
            node_status.code = resp.trainerStatus[node_id].code
            node_status.status = resp.trainerStatus[node_id].status
        except Exception as ex:
            logger.error(ex, exc_info=True)
            logger.error(f"Get {node_id} Status Error.")
            node_status.code = -1
        response.trainerStatus[node_id].CopyFrom(node_status)
    return response.trainerStatus
