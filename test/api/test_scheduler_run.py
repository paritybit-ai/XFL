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
import scheduler_run
from common.storage.redis.redis_conn import RedisConn
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode

# def test_start_server(mocker):
#     mocker.patch.object(FedConfig, 'load_config')
#     mocker.patch.object(FedNode, 'listening_port', 55001)
#     mocker.patch.object(FedJob, 'job_id', 1)
#     mocker.patch.object(FedConfig, 'trainer_config', return_value={0:'stage-1'})
#     mocker.patch('scheduler_run.trainer_control')
#     mocker.patch('scheduler_run.get_trainer_status', return_value={'node-1': status_pb2.Status(code=4, status='FAILED')})
#     mocker.patch.object(FedJob, 'status', status_pb2.TRAINING)
#     mocker.patch.object(logger, 'warning')
#     mocker.patch.object(logger, 'info')
#     mocker.patch.object(RedisConn, 'set')
#     mocker.patch('scheduler_run.remove_log_handler')

#     scheduler_run.start_server('config_path')
#     logger.info.assert_called_once_with("Stage 0 Start...")
#     scheduler_run.trainer_control.assert_called_once_with(control_pb2.START)
#     logger.warning.assert_called_once_with("Stage 0 Failed.")
#     assert FedJob.status == status_pb2.FAILED
#     scheduler_run.trainer_control.assert_called_once_with(control_pb2.STOP)
#     logger.warning.assert_called_once_with("JOB_ID: 1 Failed.")
#     RedisConn.set.assert_called_once_with("XFL_JOB_STATUS_1", status_pb2.FAILED)
#     scheduler_run.remove_log_handler.assert_called_once()

#     mocker.patch.object(FedJob, 'status', status_pb2.TRAINING)
#     mocker.patch('scheduler_run.get_trainer_status', return_value={'node-1': status_pb2.Status(code=3, status='SUCCESSFUL')})
#     mocker.patch.object(FedNode, 'trainers',{'node-1':'test'})
#     logger.info.assert_called_once_with("Stage 0 Start...")
#     scheduler_run.trainer_control.assert_called_once_with(control_pb2.START)
#     logger.warning.assert_called_once_with("Stage 0 Successful.")
#     logger.info.assert_called_once_with("All Stage Successful.")
#     logger.info.assert_called_once_with("JOB_ID: 1 Successful.")
#     RedisConn.set.assert_called_once_with("XFL_JOB_STATUS_1", status_pb2.SUCCESSFUL)
#     assert FedJob.status == status_pb2.SUCCESSFUL
#     scheduler_run.remove_log_handler.assert_called_once()


def test_main(mocker):
    mocker.patch.object(RedisConn,'init_redis')
    mocker.patch.object(FedNode, 'init_fednode')
    mocker.patch.object(FedJob, 'init_fedjob')
    mocker.patch.object(FedConfig, 'load_algorithm_list')
    mocker.patch('scheduler_run.Commu')
    mocker.patch.object(FedNode,'config', {'node-1':'test'})
    mocker.patch('scheduler_run.start_server')

    scheduler_run.main('test')
    RedisConn.init_redis.assert_called_once()
    FedNode.init_fednode.assert_called_once()
    FedJob.init_fedjob.assert_called_once()
    FedConfig.load_algorithm_list.assert_called_once()
    scheduler_run.Commu.assert_called_once_with({'node-1':'test'})
    scheduler_run.start_server.assert_called_once_with('test')




    





