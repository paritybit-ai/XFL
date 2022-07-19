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
from concurrent import futures

import grpc
import pytest

from common.communication.gRPC.python import (commu_pb2, control_pb2,
                                              scheduler_pb2,
                                              scheduler_pb2_grpc, status_pb2)
from common.storage.redis.redis_conn import RedisConn
from common.utils.grpc_channel_options import insecure_options
import service.scheduler
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.scheduler import SchedulerService

host = 'localhost'
listening_port = 55001


@pytest.fixture(scope='module', autouse=True)
def start_scheduler():
    # 启动scheduler
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=insecure_options)
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(
        SchedulerService(), server)
    server.add_insecure_port(f"[::]:{listening_port}")
    server.start()

    yield

    server.stop(None)


@pytest.fixture()
def start_client():
    channel = grpc.insecure_channel(
        f"{host}:{listening_port}", options=insecure_options)
    stub = scheduler_pb2_grpc.SchedulerStub(channel)
    return stub


def yield_post_request():
    requests = [
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(1)),
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(2)),
        commu_pb2.PostRequest(key='test~test_channel_1~1', value=bytes(3))
    ]
    for r in requests:
        yield r


class TestSchedulerService():

    def test_post(self, start_client, mocker):
        # mock redis service
        mocker.patch.object(RedisConn, 'put')
        response = start_client.post(yield_post_request())
        assert response == commu_pb2.PostResponse(code=0)
        request_key = 'test~test_channel_1~1'
        RedisConn.put.assert_called_once_with(request_key, bytes(6))

    @pytest.mark.parametrize('nodeId, config', [('node-1',  {0: {'node-1': {'trainer': 'test'}, 'node-2': {'label_trainer': 'test'}}})])
    def test_getConfig(self, start_client, nodeId, config, mocker):
        mocker.patch.object(FedConfig, 'trainer_config', config)
        mocker.patch.object(FedJob, 'current_stage', 0)
        mocker.patch.object(FedJob, 'job_id', 0)
        request = scheduler_pb2.GetConfigRequest(nodeId=nodeId)
        response = start_client.getConfig(request)
        assert response == scheduler_pb2.GetConfigResponse(
            config=json.dumps(config[0][nodeId]), code=0, jobId=0)

    def test_control(self, start_client, mocker):

        mocker.patch('service.scheduler.trainer_control',
                     return_value=control_pb2.ControlResponse(code=1, message='test'))
        mocker.patch.object(FedJob, 'job_id', 1)
        request = control_pb2.ControlRequest(control=control_pb2.STOP)
        response = start_client.control(request)
        service.scheduler.trainer_control.assert_called_once_with(
            control_pb2.STOP)
        assert response == control_pb2.ControlResponse(
            code=1, message='Stop Scheduler Successful.\n'+'test', jobId=1)

        mocker.patch.object(FedJob, 'job_id', 1)
        mocker.patch.object(FedJob, 'status', status_pb2.STOP_TRAIN)
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        assert response == control_pb2.ControlResponse(
            code=1, message="Scheduler not ready.", jobId=1)

        mocker.patch.object(FedJob, 'status', status_pb2.IDLE)
        mocker.patch('service.scheduler.get_trainer_status', return_value={
                     'node-1': status_pb2.Status(code=2, status='TRAINING')})
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        service.scheduler.get_trainer_status.assert_called()
        assert response == control_pb2.ControlResponse(
            code=1, message="Trainer node-1 not ready..", jobId=1)

        mocker.patch('service.scheduler.get_trainer_status', return_value={
                     'node-1': status_pb2.Status(code=4, status='FAILED')})
        mocker.patch.object(RedisConn, 'incr', return_value=2)
        mocker.patch.object(RedisConn, 'set')
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        RedisConn.incr.assert_called_once_with('XFL_JOB_ID')
        RedisConn.set.assert_called_once_with(
            "XFL_JOB_STATUS_2", status_pb2.TRAINING)
        assert response == control_pb2.ControlResponse(
            code=0, message="Ack", jobId=2)
        assert FedJob.status == status_pb2.TRAINING

    def test_status(self, start_client, mocker):
        # 当前节点状态
        mocker.patch.object(FedJob, 'job_id', 2)
        mocker.patch.object(FedJob, 'status', 2)
        mocker.patch('service.scheduler.get_trainer_status', return_value={
                     'node-1': status_pb2.Status(code=2, status='TRAINING')})
        request = status_pb2.StatusRequest(jobId=0)
        response = start_client.status(request)
        assert response.schedulerStatus == status_pb2.Status(
            code=2, status='TRAINING')
        service.scheduler.get_trainer_status.assert_called()
        assert response.trainerStatus == {
            'node-1': status_pb2.Status(code=2, status='TRAINING')}
        assert response.jobId == 2

        request = status_pb2.StatusRequest(jobId=2)
        response = start_client.status(request)
        assert response.jobStatus == status_pb2.Status(
            code=2, status='TRAINING')
        assert response.jobId == 2

        mocker.patch.object(
            RedisConn, 'get', return_value=status_pb2.SUCCESSFUL)
        request = status_pb2.StatusRequest(jobId=1)
        response = start_client.status(request)
        RedisConn.get.assert_called_once_with("XFL_JOB_STATUS_1")
        assert response.jobStatus == status_pb2.Status(
            code=3, status='SUCCESSFUL')

        mocker.patch.object(RedisConn, 'get', return_value=status_pb2.FAILED)
        request = status_pb2.StatusRequest(jobId=1)
        response = start_client.status(request)
        RedisConn.get.assert_called_once_with("XFL_JOB_STATUS_1")
        assert response.jobStatus == status_pb2.Status(code=4, status='FAILED')

    @pytest.mark.parametrize('algo, config',
                             [
                                 ('vertical_xgboost', {
                                  "trainer": 'test', "label_trainer": 'test'}),
                                 ('local_normalization', {
                                  "trainer": 'test', "label_trainer": 'test'})
                             ])
    def test_getAlgorithmList(self, start_client, algo, config, mocker):
        mocker.patch.object(FedConfig, 'algorithm_list', [
                            'vertical_xgboost', 'local_normalization'])
        mocker.patch.object(FedConfig, 'default_config_map', {'vertical_xgboost': {'trainer': {'info': 'test'}, 'label_trainer': {
                            'info': 'test'}}, 'local_normalization': {'trainer': {'info': 'test'}, 'label_trainer': {'info': 'test'}}})
        mocker.patch.object(json, 'dumps', return_value='test')
        request = scheduler_pb2.GetAlgorithmListRequest()
        response = start_client.getAlgorithmList(request)
        assert response.algorithmList == [
            'vertical_xgboost', 'local_normalization']
        assert response.defaultConfigMap[algo] == scheduler_pb2.DefaultConfig(
            config=config)
