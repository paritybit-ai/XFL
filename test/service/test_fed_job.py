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


from common.storage.redis.redis_conn import RedisConn
from common.communication.gRPC.python import status_pb2
from service.fed_job import FedJob


class mock_model(object):
    def __init__(self, stage_config):
        pass


class Test_FedJob():

    def test_init_fedjob(self, mocker):
        mocker.patch.object(RedisConn, 'get', return_value=1)
        FedJob.init_fedjob()
        RedisConn.get.assert_called_once_with("XFL_JOB_ID")
        assert FedJob.job_id == 1

    def test_init_progress(self):
        FedJob.init_progress(2)
        assert FedJob.total_stage_num == 2
        assert FedJob.progress == [0, 0]
    
    def test_get_model(self, mocker):
        # mock get_operator
        mocker.patch('service.fed_job.get_operator', return_value=mock_model)
        model = FedJob.get_model("trainer", {"model_info": {"name": "VerticalKmeansTrainer"}})
        assert isinstance(model, mock_model)
        