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


import pytest

import client
from common.communication.gRPC.python import (control_pb2, scheduler_pb2,
                                              status_pb2)
from common.storage.redis.redis_conn import RedisConn
from service.fed_node import FedNode


def test_start(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='55001')
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.__init__", side_effect=lambda x:None)
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.control", create=True, return_value=control_pb2.ControlResponse(jobId=1,code=1, message='test'))
    client.start()
    client.scheduler_pb2_grpc.SchedulerStub.control.assert_called_once_with(control_pb2.ControlRequest(control = control_pb2.START))


def test_stop(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='55001')
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.__init__", side_effect=lambda x:None)
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.control", create=True, return_value=control_pb2.ControlResponse(jobId=1,code=1, message='test'))
    client.stop()
    client.scheduler_pb2_grpc.SchedulerStub.control.assert_called_once_with(control_pb2.ControlRequest(control = control_pb2.STOP))



def test_status(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='55001')
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.__init__", side_effect=lambda x:None)
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.status", create=True, return_value=status_pb2.StatusResponse(jobId=0,schedulerStatus=status_pb2.Status(code=1,status='IDLE'),trainerStatus={"node-1":status_pb2.Status(code=1,status='IDLE')}))
    client.status()
    client.scheduler_pb2_grpc.SchedulerStub.status.assert_called_once_with(status_pb2.StatusRequest())


def test_algo(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='55001')
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.__init__", side_effect=lambda x:None)
    mocker.patch("client.scheduler_pb2_grpc.SchedulerStub.getAlgorithmList", create=True, return_value=scheduler_pb2.GetAlgorithmListResponse(algorithmList=['test'],defaultConfigMap={"test_map":scheduler_pb2.DefaultConfig(config={"test_k":"test_v"})}))
    client.algo()
    client.scheduler_pb2_grpc.SchedulerStub.getAlgorithmList.assert_called_once_with(scheduler_pb2.GetAlgorithmListRequest())


@pytest.mark.parametrize('cmd',
                             [
                                 ('start'),
                                 ('stop'),
                                 ('status'),
                                 ('algo')
                             ])
def test_main(mocker,cmd):
    mocker.patch.object(RedisConn, "init_redis")
    mocker.patch.object(FedNode, "init_fednode")
    mocker.patch.object(client, cmd)
    client.main(cmd)
    getattr(client,cmd).assert_called()

