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
from common.communication.gRPC.python import scheduler_pb2

import service
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode


class Test_FedConfig():
    @pytest.mark.parametrize('trainer_list, result',
                             [
                                 (['test_1', 'test_2'], ['test_1', 'test_2']),
                                 ([], [])
                             ])
    def test_get_label_trainer(self, mocker, trainer_list, result):
        mocker.patch.object(FedConfig, 'stage_config', {
            "fed_info": {"label_trainer": trainer_list}})
        res = FedConfig.get_label_trainer()
        assert res == result

    @pytest.mark.parametrize('trainer_list, result',
                             [
                                 (['test_1', 'test_2'], 'test_1'),
                                 ([], None)
                             ])
    def test_get_assist_trainer(self, mocker, trainer_list, result):
        mocker.patch.object(FedConfig, 'stage_config', {
            "fed_info": {"assist_trainer": trainer_list}})
        res = FedConfig.get_assist_trainer()
        assert res == result

    @pytest.mark.parametrize('trainer_list, result',
                             [
                                 (['test_1', 'test_2'], ['test_1', 'test_2']),
                                 ([], [])
                             ])
    def test_get_trainer(self, mocker, trainer_list, result):
        mocker.patch.object(FedConfig, 'stage_config', {
            "fed_info": {"trainer": trainer_list}})
        res = FedConfig.get_trainer()
        assert res == result

    def test_load_config(self, mocker):
        mocker.patch.object(FedJob, 'job_id', 1)
        mocker.patch('service.fed_config.add_job_log_handler')
        mocker.patch.object(FedConfig, 'load_trainer_config', return_value={})
        FedConfig.load_config('test')
        service.fed_config.add_job_log_handler.assert_called_once_with(1, '')
        assert FedConfig.trainer_config == {}

    def test_load_trainer_config(self, mocker):
        mocker.patch.object(FedNode, 'trainers', {"node-1": "test"})
        mocker.patch('service.fed_config.load_json_config',
                     return_value=[{"identity": "trainer"}])
        mocker.patch("os.path.exists", return_value=True)

        trainer_config = FedConfig.load_trainer_config("test")

        assert trainer_config == {0: {"node-1": {"identity": "trainer", "fed_info": {
            "label_trainer": [],
            "trainer": ["node-1"],
            "assist_trainer": []
        }}}}

        ##

        mocker.patch.object(FedNode, 'trainers', {"node-1": "test", "assist_trainer": "test2"})

        def mock_load_json_config(*args, **kwargs):
            if load_json_config.call_count == 1:
                return [{
                    "identity": "trainer",
                    "model_info": {
                        "name": "vertical_xgboost"
                    }
                }]
            elif load_json_config.call_count == 2:
                return [{}]

        load_json_config = mocker.patch('service.fed_config.load_json_config', side_effect=mock_load_json_config)

        def mock_func(*args, **kwargs):
            if os_path.call_count % 2 == 1:
                return True
            elif os_path.call_count % 2 == 0:
                return False

        os_path = mocker.patch("os.path.exists", side_effect=mock_func)

        trainer_config = FedConfig.load_trainer_config("test")

        assert trainer_config == \
               {
                   0: {
                       'node-1': {
                           'identity': 'trainer',
                           'model_info': {
                               'name': 'vertical_xgboost'
                           },
                           'fed_info': {
                               'label_trainer': [],
                               'trainer': ['node-1'],
                               'assist_trainer': []
                           }
                       },
                       'assist_trainer': {
                           'fed_info': {
                               'label_trainer': [],
                               'trainer': ['node-1'],
                               'assist_trainer': []
                           }
                       }
                   }
               }

        ##
        def mock_func(*args, **kwargs):
            return False

        os_path = mocker.patch("os.path.exists", side_effect=mock_func)

        trainer_config = FedConfig.load_trainer_config("test")
        assert trainer_config == {}

        ##
        mocker.patch.object(FedNode, 'trainers', {"node-1": "test", "node-2": "test3", "assist_trainer": "test2"})

        def mock_load_json_config(*args, **kwargs):
            if load_json_config.call_count == 1:
                return [{
                    "identity": "trainer",
                    "model_info": {
                        "name": "vertical_xgboost"
                    }
                }]
            elif load_json_config.call_count == 2:
                return [
                    {"identity": "label_trainer",
                     "model_info": {
                         "name": "vertical_logistic_regression"
                     }
                     }
                ]
            else:
                return [{
                    "identity": "assist_trainer",
                    "model_info": {
                        "name": "vertical_xgboost"
                    }
                }]

        load_json_config = mocker.patch('service.fed_config.load_json_config', side_effect=mock_load_json_config)

        def mock_func(*args, **kwargs):
            if os_path.call_count <= 2:
                return True
            else:
                return False

        os_path = mocker.patch("os.path.exists", side_effect=mock_func)
        trainer_config = FedConfig.load_trainer_config("test")
        assert trainer_config == \
               {
                   0:
                       {
                           'node-1': {
                               'identity': 'trainer',
                               'model_info': {'name': 'vertical_xgboost'},
                               'fed_info': {'label_trainer': ['node-2'], 'trainer': ['node-1'], 'assist_trainer': []}
                           },
                           'node-2': {
                               'identity': 'label_trainer',
                               'model_info': {'name': 'vertical_logistic_regression'},
                               'fed_info': {'label_trainer': ['node-2'], 'trainer': ['node-1'], 'assist_trainer': []}
                           },
                       }
               }

        ###
        ##
        mocker.patch.object(FedNode, 'trainers', {"node-1": "test", "node-2": "test3", "assist_trainer": "test2"})

        def mock_load_json_config(*args, **kwargs):
            if load_json_config.call_count == 1:
                return [{
                    "identity": "trainer",
                    "model_info": {
                        "name": "vertical_kmeans"
                    }
                }]
            elif load_json_config.call_count == 2:
                return [{
                    "identity": "label_trainer",
                    "model_info": {
                        "name": "vertical_kmeans"
                    }
                }]
            else:
                return [{}]

        load_json_config = mocker.patch('service.fed_config.load_json_config', side_effect=mock_load_json_config)

        def mock_func(*args, **kwargs):
            if os_path.call_count <= 2:
                return True
            else:
                return False

        os_path = mocker.patch("os.path.exists", side_effect=mock_func)
        trainer_config = FedConfig.load_trainer_config("test")
        # assert trainer_config == \
        #     {
        #         0: 
        #             {
        #                 'node-1': {
        #                     'identity': 'trainer', 
        #                     'model_info': {'name': 'vertical_xgboost'}, 
        #                     'fed_info':  {'label_trainer': ['node-2'], 'trainer': ['node-1'], 'assist_trainer': []}
        #                 }, 
        #                 'node-2': {
        #                     'identity': 'label_trainer', 
        #                     'model_info': {'name': 'vertical_logistic_regression'}, 
        #                     'fed_info': {'label_trainer': ['node-2'], 'trainer': ['node-1'], 'assist_trainer': []}
        #                 },
        #             }
        #     }

    def test_get_config(self, mocker):

        mocker.patch.object(FedNode, "create_channel", return_value='55001')
        mocker.patch("service.fed_config.scheduler_pb2_grpc.SchedulerStub.__init__", side_effect=lambda x: None)
        mocker.patch("service.fed_config.scheduler_pb2_grpc.SchedulerStub.getConfig", create=True,
                     return_value=scheduler_pb2.GetConfigResponse(jobId=2, config="test_config"))

        mocker.patch.object(FedJob, "global_epoch", 0)
        mocker.patch("json.loads",
                     return_value={"model_info": {"name": "test"}, "train_info": {"train_params": {"global_epoch": 1}}})
        mocker.patch("service.fed_config.add_job_log_handler", return_value="job_log_handler")
        mocker.patch("service.fed_config.add_job_stage_log_handler", return_value="job_stage_log_handler")

        resp = FedConfig.get_config()

        FedNode.create_channel.assert_called_once_with("scheduler")
        assert FedConfig.job_log_handler == "job_log_handler"
        assert FedConfig.job_stage_log_handler == "job_stage_log_handler"
        service.fed_config.add_job_log_handler.assert_called_once_with(2, '')
        service.fed_config.add_job_stage_log_handler.assert_called_once_with(2, '', 0, "test")
        assert FedJob.global_epoch == 1
        assert resp.config == "test_config"

    def test_load_algorithm_list(self, mocker):
        def mock_load_json_config(args):
            if '/algorithm/config/vertical_xgboost/trainer' in args:
                return {"identity": "trainer"}
            elif '/algorithm/config/vertical_xgboost/label_trainer' in args:
                return {"identity": "label_trainer"}

        mocker.patch('service.fed_config.load_json_config',
                     side_effect=mock_load_json_config)
        FedConfig.load_algorithm_list()

        assert FedConfig.default_config_map["vertical_xgboost"] == {"trainer": {
            "identity": "trainer"}, "label_trainer": {"identity": "label_trainer"}}
