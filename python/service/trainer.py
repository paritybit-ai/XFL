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


from common.communication.gRPC.python import commu_pb2, control_pb2, status_pb2
from common.storage.redis.redis_conn import RedisConn
# from common.utils.logger import logger
from service.fed_job import FedJob
from service.fed_node import FedNode


class TrainerService(object):
    def __init__(self):
        pass
    
    def post(self, request, context):
        request_key = ''
        request_value = bytearray()
        
        for i, r in enumerate(request):
            request_value += r.value
            if i == 0:
                request_key = r.key
                # request_info = r.key.split("~")
                # name = request_info[1]
                # start_end_id = request_info[-1]
                # logger.info(f"Start receiving the data of channel {name} from {start_end_id} ...")
            
        RedisConn.put(request_key, bytes(request_value))
        # logger.info(f"Successfully received the data of channel {name} from {start_end_id}")
        response = commu_pb2.PostResponse()
        response.code = 0
        return response

    def control(self, request, context):
        response = control_pb2.ControlResponse()
        if request.control == control_pb2.STOP:
            FedJob.status.value = status_pb2.STOP_TRAIN
        elif request.control == control_pb2.START:
            FedJob.status.value = status_pb2.START_TRAIN
        response.code = 0
        response.message = f"{status_pb2.StatusEnum.Name(request.control)} Completed."
        return response

    def status(self, request, context):
        response = status_pb2.StatusResponse()
        node_status = status_pb2.Status()
        node_status.code = FedJob.status.value
        node_status.status = status_pb2.StatusEnum.Name(FedJob.status.value)
        response.trainerStatus[FedNode.node_id].CopyFrom(node_status)
        return response
