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


from algorithm.framework.vertical.kmeans.trainer import VerticalKmeansTrainer
from common.storage.redis.redis_conn import RedisConn

from service.fed_job import FedJob


class mock_model():
    VerticalKmeansTrainer = lambda x,y: x

class Test_FedJob():

    def test_init_fedjob(self, mocker):

        mocker.patch.object(RedisConn, 'get', return_value=1)
        FedJob.init_fedjob()
        RedisConn.get.assert_called_once_with("XFL_JOB_ID")
        assert FedJob.job_id == 1

    # def test_get_model(self, mocker):
    #     # mocker.patch("service.fed_job.load_json_config", return_value={"vertical_kmeans": {
    #     #     "assist_trainer": {
    #     #         "module_name": "algorithm.framework.vertical.kmeans.assist_trainer",
    #     #         "class_name": "VerticalKmeansAssist_trainer"
    #     #     },
    #     #     "label_trainer": {
    #     #         "module_name": "algorithm.framework.vertical.kmeans.trainer",
    #     #         "class_name": "VerticalKmeansTrainer"
    #     #     },
    #     #     "trainer": {
    #     #         "module_name": "algorithm.framework.vertical.kmeans.trainer",
    #     #         "class_name": "VerticalKmeansTrainer"}}})
    #     # mocker.patch.object(importlib, "import_module", return_value=mock_model)
    
    #     stage_config = {"model_info": {"name": "vertical_kmeans","config":{"test":{}}}, "train_info": {"params": {}}, "input": {}, "output": {}}
    #     model = FedJob.get_model("trainer", stage_config)

    #     importlib.import_module.assert_called_once_with(
    #         "algorithm.framework.vertical.kmeans.vertical_kmeans_trainer")
        
    #     assert model == stage_config 

        
