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
from unittest.mock import call
import scheduler_run
from common.storage.redis.redis_conn import RedisConn
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode
from common.utils.logger import logger
from common.communication.gRPC.python import status_pb2

def test_start_server(mocker):
    mocker.patch.object(FedConfig, 'load_config')
    mocker.patch.object(FedNode, 'listening_port', 55001)
    mocker.patch.object(FedJob, 'job_id', 1)
    mocker.patch.object(FedConfig, 'trainer_config', {0: {'node-1': {}}})
    mocker.patch('scheduler_run.trainer_control')
    mocker.patch('scheduler_run.get_trainer_status', return_value={'node-1': status_pb2.Status(code=4, status='FAILED')})
    mocker.patch.object(FedJob, 'status', status_pb2.TRAINING)
    mocker.patch.object(logger, 'warning')
    mocker.patch.object(logger, 'info')
    mocker.patch.object(RedisConn, 'set')
    mocker.patch('scheduler_run.remove_log_handler', side_effect=RuntimeError())

    with pytest.raises(RuntimeError) as e:
        scheduler_run.start_server('config_path', is_bar=True)

    logger.info.assert_called()
    scheduler_run.trainer_control.assert_called()
    logger.warning.assert_called()
    assert FedJob.status == status_pb2.FAILED
    scheduler_run.trainer_control.assert_called()
    # RedisConn.set.assert_called_once_with("XFL_JOB_STATUS_1", status_pb2.FAILED)
    # RedisConn.set.assert_called_with("XFL_JOB_STATUS_1", status_pb2.FAILED)
    # calls = [
    #     call("XFL_JOB_STATUS_1",), call("XFL_JOB_START_TIME_1",), call("XFL_JOB_END_TIME_1",)
    # ]
    # RedisConn.set.assert_has_calls(calls)
    
    scheduler_run.remove_log_handler.assert_called()



def test_main(mocker):
    mocker.patch.object(RedisConn,'init_redis')
    mocker.patch.object(FedNode, 'init_fednode')
    mocker.patch.object(FedJob, 'init_fedjob')
    mocker.patch.object(FedConfig, 'load_algorithm_list')
    mocker.patch('scheduler_run.Commu')
    mocker.patch.object(FedNode,'config', {'node-1':'test'})
    mocker.patch('scheduler_run.start_server')

    scheduler_run.main('test', is_bar=True)
    RedisConn.init_redis.assert_called_once()
    FedNode.init_fednode.assert_called_once()
    FedJob.init_fedjob.assert_called_once()
    FedConfig.load_algorithm_list.assert_called_once()
    scheduler_run.Commu.assert_called_once_with({'node-1':'test'})
    scheduler_run.start_server.assert_called_once_with('test', True)




    





