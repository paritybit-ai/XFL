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


from common.communication.gRPC.python import (control_pb2, scheduler_pb2,
                                              scheduler_pb2_grpc, status_pb2)
from common.storage.redis.redis_conn import RedisConn
from service.fed_node import FedNode


def start():
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = control_pb2.ControlRequest()
    request.control = control_pb2.START
    response = stub.control(request)
    print("JobID:", response.jobId)
    print("Code:", response.code)
    print("Message:", response.message)


def stop():
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = control_pb2.ControlRequest()
    request.control = control_pb2.STOP
    response = stub.control(request)
    print("Code:", response.code)
    print("JobID:", response.jobId)
    print("Message:", response.message)


def status():
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = status_pb2.StatusRequest()
    response = stub.status(request)
    print("JobID:", response.jobId)
    if request.jobId == 0:
        print("---------- Scheduler ----------")
        print("Code:", response.schedulerStatus.code)
        print("Status:",  response.schedulerStatus.status)
        for node_id in response.trainerStatus.keys():
            print(f"---------- Trainer {node_id} ----------")
            print("Code:", response.trainerStatus[node_id].code)
            print("Status:", response.trainerStatus[node_id].status)
    else:
        print("Code:", response.jobStatus.code)
        print("Job Status:",  response.jobStatus.status)


def algo():
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = scheduler_pb2.GetAlgorithmListRequest()
    response = stub.getAlgorithmList(request)
    print("---------- Algorithm List ----------")
    for i in response.algorithmList:
        print(i)
    print(f"---------- Config ----------")
    print({i: response.defaultConfigMap[i].config for i in response.algorithmList})


def stage():
    channel = FedNode.create_channel("scheduler")
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    request = scheduler_pb2.GetStageRequest()
    response = stub.getStage(request)
    print("---------- Stage ----------")
    print("code:", response.code)
    print("stage_id:", response.stageId)
    print("stage_name:", response.stageName)


def main(cmd, config_path=''):
    FedNode.init_fednode(conf_dir=config_path)
    RedisConn.init_redis()
    if cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    elif cmd == "status":
        status()
    elif cmd == "algo":
        algo()
    elif cmd == "stage":
        stage()
    else:
        print("Client command is not exists.")
