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
from unittest.mock import call

import trainer_run
from common.communication.gRPC.python import control_pb2, status_pb2
from common.storage.redis.redis_conn import RedisConn
from common.utils.logger import logger
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode


class mock_model():
    def fit(self):
        pass

class mock_server():
    def start(self):
        pass
    def wait_for_termination(self):
        pass



# @pytest.fixture()
# def start_stub():
#     channel = grpc.insecure_channel(
#         f"localhost:56001", options=insecure_options)
#     stub = trainer_pb2_grpc.TrainerStub(channel)
#     return stub


def test_start_trainer_service(mocker):
    from unittest.mock import call
    mocker.patch.object(trainer_run.grpc, 'server', return_value = mock_server())
    mocker.patch.object(FedJob,'status')
    mocker.patch("trainer_run.trainer_pb2_grpc.add_TrainerServicer_to_server")
    mocker.patch.object(FedNode, 'add_server')
    mocker.patch.object(logger, 'info')
    mocker.patch.object(FedNode, 'listening_port', 56001)
    trainer_run.start_trainer_service(status_pb2.IDLE)

    trainer_run.trainer_pb2_grpc.add_TrainerServicer_to_server.assert_called()
    FedNode.add_server.assert_called()
    logger.info.assert_has_calls([call("Trainer Service Start..."),call("[::]:56001")])
    assert FedJob.status == status_pb2.IDLE


# def test_start_server(start_stub, mocker):
#     mocker.patch.object(multiprocessing,"Process",return_value=mock_Process())
#     mocker.patch.object(logger,"info")
#     mocker.patch("trainer_run.remove_log_handler")
#     trainer_run.start_server()
#     response = start_stub.control(control_pb2.ControlRequest(control=control_pb2.START))

#     assert FedJob.status.value == status_pb2.TRAINING
#     response = start_stub.control(control_pb2.ControlRequest(control=control_pb2.STOP))

#     logger.info.assert_called_once_with("Model training is stopped.")
#     assert FedJob.status.value == status_pb2.FAILED
#     trainer_run.remove_log_handler.assert_called()

def test_train(mocker):
    
    mocker.patch.object(FedJob, "get_model", return_value=mock_model())
    mocker.patch.object(FedConfig, 'get_config')
    mocker.patch.object(FedConfig, 'stage_config', {"identity": "node-1"})
    mocker.patch.object(logger, 'info')

    status = multiprocessing.Value("i", status_pb2.IDLE)
    trainer_run.train(status)
    assert status.value == status_pb2.SUCCESSFUL
    FedJob.get_model.assert_called_once_with("node-1", {"identity": "node-1"})
    logger.info.assert_has_calls(
        [call("node-1 Start Training..."), call("Train Model Successful.")])


def test_job_control(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='55001')
    mocker.patch("trainer_run.scheduler_pb2_grpc.SchedulerStub.__init__",
                 side_effect=lambda x: None)
    mocker.patch("trainer_run.scheduler_pb2_grpc.SchedulerStub.control", create=True,
                 return_value=control_pb2.ControlResponse(code=0, message='test'))
    mocker.patch.object(logger, 'info')
    trainer_run.job_control(1)
    trainer_run.scheduler_pb2_grpc.SchedulerStub.control.assert_called_once_with(
        control_pb2.ControlRequest(control=1))
    logger.info.assert_called_once_with(
        control_pb2.ControlResponse(code=0, message='test'))


def test_main(mocker):
    mocker.patch.object(RedisConn, 'init_redis')
    mocker.patch.object(FedNode, 'init_fednode')
    mocker.patch.object(FedNode, 'config', {})
    mocker.patch('trainer_run.Commu')
    mocker.patch('trainer_run.start_server')

    trainer_run.main('trainer', 'node-1')
    RedisConn.init_redis.assert_called_once()
    FedNode.init_fednode.assert_called_once_with(
        identity='trainer', debug_node_id='node-1', conf_dir='')
    trainer_run.Commu.assert_called_once_with({})
    trainer_run.start_server.assert_called()
