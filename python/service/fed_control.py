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


from common.communication.gRPC.python import control_pb2, status_pb2, trainer_pb2_grpc
from common.utils.logger import logger

from service.fed_node import FedNode


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
