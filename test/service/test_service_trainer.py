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


from concurrent import futures

import grpc
import pytest
from common.communication.gRPC.python import (commu_pb2, control_pb2,
                                              status_pb2, trainer_pb2_grpc)
from common.storage.redis.redis_conn import RedisConn
from common.utils.grpc_channel_options import insecure_options
from service.fed_job import FedJob
from service.fed_node import FedNode
from service.trainer import TrainerService

host = 'localhost'
listening_port = 56001


@pytest.fixture(scope='module', autouse=True)
def start_trainer():
    # 启动scheduler
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=insecure_options)
    trainer_pb2_grpc.add_TrainerServicer_to_server(
        TrainerService(), server)
    server.add_insecure_port(f"[::]:{listening_port}")
    server.start()

    yield

    server.stop(None)


@pytest.fixture()
def start_client():
    channel = grpc.insecure_channel(
        f"{host}:{listening_port}", options=insecure_options)
    stub = trainer_pb2_grpc.TrainerStub(channel)
    return stub


def yield_post_request():
    requests = [
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(1)),
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(2)),
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(3))
    ]
    for r in requests:
        yield r


class TestTrainerService():
    def test_post(self, start_client, mocker):
        # mock redis service
        mocker.patch.object(RedisConn, 'put')

        response = start_client.post(yield_post_request())
        assert response == commu_pb2.PostResponse(code=0)
        request_key = 'test~test_channel_1~1'
        RedisConn.put.assert_called_once_with(request_key, bytes(6))

    @pytest.mark.parametrize('action',
                             [(control_pb2.START),
                              (control_pb2.STOP),
                              ])
    def test_control(self, start_client, action, mocker):
        def action2status(action):
            if action == control_pb2.START:
                return status_pb2.START_TRAIN
            if action == control_pb2.STOP:
                return status_pb2.STOP_TRAIN
        mocker.patch.object(FedJob, 'status', spec='value', create=True)
        request = control_pb2.ControlRequest(control=action)
        response = start_client.control(request)
        assert response.code == 0
        assert FedJob.status.value == action2status(action)
        assert response.message == f"{status_pb2.StatusEnum.Name(request.control)} Completed."

    @pytest.mark.parametrize('node_id, code',
                             [('node-1', 1),
                              ('node-2', 2),
                                 ('node-3', 3),
                              ])
    def test_status(self, start_client, node_id, code, mocker):
        mocker.patch.object(FedJob, 'status', spec='value', create=True)
        mocker.patch.object(FedNode, 'node_id', node_id)
        FedJob.status.value = code
        request = status_pb2.StatusRequest()
        response = start_client.status(request)
        assert response.trainerStatus[node_id] == status_pb2.Status(
            code=code, status=status_pb2.StatusEnum.Name(code))
