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
from google.protobuf import text_format
from google.protobuf import json_format

import service.scheduler
from common.communication.gRPC.python import (checker_pb2, commu_pb2,
                                              control_pb2, scheduler_pb2,
                                              scheduler_pb2_grpc, status_pb2)
from common.storage.redis.redis_conn import RedisConn
from common.utils.config_parser import replace_variable
from common.utils.grpc_channel_options import insecure_options
from common.utils.logger import get_node_log_path, get_stage_node_log_path
from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode
from service.scheduler import SchedulerService

host = 'localhost'
listening_port = 55001


@pytest.fixture(scope='module', autouse=True)
def start_scheduler():
    # 启动scheduler
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=insecure_options)
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(
        SchedulerService(is_bar=True), server)
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
            code=1, message='Stop Scheduler Successful.\n'+'test', jobId=1, nodeLogPath={}, stageNodeLogPath={})

        mocker.patch.object(FedJob, 'job_id', 1)
        mocker.patch.object(FedJob, 'status', status_pb2.STOP_TRAIN)
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        assert response == control_pb2.ControlResponse(
            code=1, message="Scheduler not ready.", jobId=1, nodeLogPath={}, stageNodeLogPath={})

        mocker.patch.object(FedJob, 'status', status_pb2.IDLE)
        mocker.patch('service.scheduler.get_trainer_status', return_value={
                     'node-1': status_pb2.Status(code=2, status='TRAINING')})
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        service.scheduler.get_trainer_status.assert_called()
        assert response == control_pb2.ControlResponse(
            code=1, message="Trainer node-1 not ready..", jobId=1, nodeLogPath={}, stageNodeLogPath={})

        mocker.patch('service.scheduler.get_trainer_status', return_value={
                     'node-1': status_pb2.Status(code=4, status='FAILED')})
        mocker.patch.object(RedisConn, 'incr', return_value=2)
        mocker.patch.object(RedisConn, 'set')
        request = control_pb2.ControlRequest(control=control_pb2.START)
        response = start_client.control(request)
        RedisConn.incr.assert_called_once_with('XFL_JOB_ID')
        RedisConn.set.assert_called_once_with(
            "XFL_JOB_STATUS_2", status_pb2.TRAINING)
        
        job_log_path = get_node_log_path(job_id=FedJob.job_id, node_ids=list(FedNode.trainers.keys()) + ['scheduler'])
        job_stages_log_path = get_stage_node_log_path(job_id=FedJob.job_id, train_conf=FedConfig.converted_trainer_config)
        
        # if not FedConfig.trainer_config:
        #     job_log_path = {}
        #     job_stages_log_path = {}
                    
        # assert response == control_pb2.ControlResponse(
        #     code=0, message="", jobId=2, nodeLogPath=json.dumps(job_log_path), stageNodeLogPath=json.dumps(job_stages_log_path))
        # assert FedJob.status == status_pb2.TRAINING

    def test_recProgress(self, start_client, mocker):
        # mocker.patch.object(FedJob, 'progress', {0: 0})
        # request = scheduler_pb2.RecProgressRequest(stageId=0, progress=10)
        # response = start_client.recProgress(request)
        # assert response == scheduler_pb2.RecProgressResponse(code=0)
        # assert FedJob.progress[0] == 10
        mocker.patch.object(FedJob, 'job_id', 2)
        mocker.patch.object(FedJob, 'current_stage', 0)
        mocker.patch.object(FedJob, 'total_stage_num', 1)
        mocker.patch.object(FedJob, 'progress', {0: 0})
        mocker.patch.object(FedConfig, 'trainer_config', {
                            0: {'trainer': {'model_info': {'name': 'test'}}}})
        mocker.patch.object(RedisConn, 'set', return_value=None)
        request = scheduler_pb2.RecProgressRequest(stageId=0, progress=10)
        response = start_client.recProgress(request)
        assert response == scheduler_pb2.RecProgressResponse(code=0)
        assert FedJob.progress[0] == 10

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

        # request = status_pb2.StatusRequest(jobId=2)
        # response = start_client.status(request)
        # assert response.jobStatus == status_pb2.Status(
        #     code=2, status='TRAINING')
        # assert response.jobId == 2

        mocker.patch.object(
            RedisConn, 'get', return_value=status_pb2.SUCCESSFUL)
        request = status_pb2.StatusRequest(jobId=1)
        response = start_client.status(request)
        # RedisConn.get.assert_called_once_with("XFL_JOB_STATUS_1")
        assert response.jobStatus == status_pb2.Status(
            code=3, status='SUCCESSFUL')

        mocker.patch.object(RedisConn, 'get', return_value=status_pb2.FAILED)
        request = status_pb2.StatusRequest(jobId=1)
        response = start_client.status(request)
        # RedisConn.get.assert_called_once_with("XFL_JOB_STATUS_1")
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

    def test_getStage(self, start_client, mocker):
        mocker.patch.object(FedJob, 'current_stage', 2)
        mocker.patch.object(FedJob, 'total_stage_num', 3)
        progress = {0: 100, 1: 45}
        mocker.patch.object(FedJob, 'progress', progress)
        
        stage_response = scheduler_pb2.GetStageResponse()
        stage_name = "test"
        stage_response.code = 0
        stage_response.currentStageId = 1
        stage_response.totalStageNum = 3
        stage_response.currentStageName = stage_name
        
        bar_response = scheduler_pb2.ProgressBar()
        for stage, progress in progress.items():
            bar_response.stageId = stage
            bar_response.stageProgress = progress
            stage_response.progressBar.append(bar_response)
            
        mocker.patch.object(RedisConn, 'get', return_value=json_format.MessageToJson(stage_response))
        
        request = scheduler_pb2.GetStageRequest()
        request.jobId = 0
        response = start_client.getStage(request)
        assert response.code == 0
        assert response.currentStageId == 1
        assert response.totalStageNum == 3
        assert response.currentStageName == 'test'
        assert response.progressBar[0].stageId == 0
        assert response.progressBar[0].stageProgress == 100
        assert response.progressBar[1].stageId == 1
        assert response.progressBar[1].stageProgress == 45
        
        mocker.patch.object(RedisConn, 'get', return_value=None)
        request = scheduler_pb2.GetStageRequest()
        request.jobId = 0
        response = start_client.getStage(request)
        assert response.code == 3

        # mocker.patch.object(FedJob, 'current_stage', 0)
        # mocker.patch.object(FedJob, 'total_stage_num', 1)
        # mocker.patch.object(FedJob, 'progress', {0: 0})
        # mocker.patch.object(FedConfig, 'trainer_config', {
        #                     0: {'trainer': {'model_info': {'name': 'test'}}}})
        # request = scheduler_pb2.GetStageRequest()
        # response = start_client.getStage(request)
        # assert response.code == 0
        # assert response.currentStageId == 0
        # assert response.totalStageNum == 1
        # assert response.currentStageName == 'test'
        # assert response.progressBar[0].stageId == 0
        # assert response.progressBar[0].stageProgress == 0

        # mocker.patch.object(FedConfig, 'trainer_config', {0: {}})
        # request = scheduler_pb2.GetStageRequest()
        # response = start_client.getStage(request)
        # assert response.code == 1
        # assert response.currentStageName == ''

        # mocker.patch.object(FedConfig, 'trainer_config', [])
        # request = scheduler_pb2.GetStageRequest()
        # response = start_client.getStage(request)
        # assert response.code == 2
        # assert response.currentStageName == ''

    def test_checkTaskConfig(self, start_client, mocker):
        request = checker_pb2.CheckTaskConfigRequest()
        
        conf = \
        [
            {
                "identity": "label_trainer",
                "model_info": {
                    # "name": "vertical_binning_woe_iv_fintech"
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True,
                            "nan_list": [
                            ]
                        }
                    ]
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "vertical_binning_woe_iv_[STAGE_ID].json"
                    },
                    "iv": {
                        "name": "woe_iv_result_[STAGE_ID].json"
                    },
                    "split_points": {
                        "name": "binning_split_points_[STAGE_ID].json"
                    },
                    "trainset": {
                        "name": "fintech_woe_map_train_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "interaction_params": {
                        "save_model": True
                    },
                    "train_params": {
                        "encryption": {
                            "paillier": {
                                "key_bit_size": 2048,
                                "precision": 7,
                                "djn_on": True,
                                "parallelize_on": True
                            }
                        },
                        "binning": {
                            "method": "equal_width",
                            "bins": 5
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    # "name": "vertical_feature_selection"
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "iv_result": {
                        "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "woe_iv_result_[STAGE_ID-1].json"
                    },
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "feature_selection_[STAGE_ID].pkl"
                    },
                    "trainset": {
                        "name": "selected_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "selected_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "filter": {
                            "common": {
                                "metrics": "iv",
                                "filter_method": "threshold",
                                "threshold": 0.01
                            }
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    # "name": "vertical_pearson"
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "corr": {
                        "name": "vertical_pearson_[STAGE_ID].pkl"
                    }
                },
                "train_info": {
                    "train_params": {
                        "col_index": -1,
                        "col_names": "",
                        "encryption": {
                            "paillier": {
                                "key_bit_size": 2048,
                                "precision": 6,
                                "djn_on": True,
                                "parallelize_on": True
                            }
                        },
                        "max_num_cores": 999,
                        "sample_size": 9999
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    # "name": "vertical_feature_selection"
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "corr_result": {
                        "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "vertical_pearson_[STAGE_ID-1].pkl"
                    },
                    "iv_result": {
                        "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "woe_iv_result_[STAGE_ID].json" # "name": "woe_iv_result_[STAGE_ID-3].json"
                    },
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-2].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_val_[STAGE_ID-2].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "feature_selection_[STAGE_ID].pkl"
                    },
                    "trainset": {
                        "name": "selected_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "selected_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "filter": {
                            "common": {
                                "metrics": "iv",
                                "filter_method": "threshold",
                                "threshold": 0.01
                            },
                            "correlation": {
                                "sort_metric": "iv",
                                "correlation_threshold": 0.7
                            }
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    # "name": "local_normalization"
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_val_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        # "name": "local_normalization_[STAGE_ID].pt"
                        "name": "vertical_logitstic_regression_[STAGE_ID].pt"
                    },
                    "trainset": {
                        "name": "normalized_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "normalized_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "norm": "max",
                        "axis": 0
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "normalized_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "normalized_val_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "pretrained_model": {
                        "path": "",
                        "name": ""
                    }
                },
                "output": {
                    "path": "/tmp/xfl/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        # "name": "vertical_logitstic_regression_[STAGE_ID].pt"
                        "name": "vertical_logitstic_regression_[STAGE_ID - 1].pt"
                    },
                    "metric_train": {
                        "name": "lr_metric_train_[STAGE_ID].csv"
                    },
                    "metric_val": {
                        "name": "lr_metric_val_[STAGE_ID].csv"
                    },
                    "prediction_train": {
                        "name": "lr_prediction_train_[STAGE_ID].csv"
                    },
                    "prediction_val": {
                        "name": "lr_prediction_val_[STAGE_ID].csv"
                    },
                    "ks_plot_train": {
                        "name": "lr_ks_plot_train_[STAGE_ID].csv"
                    },
                    "ks_plot_val": {
                        "name": "lr_ks_plot_val_[STAGE_ID].csv"
                    },
                    "decision_table_train": {
                        "name": "lr_decision_table_train_[STAGE_ID].csv"
                    },
                    "decision_table_val": {
                        "name": "lr_decision_table_val_[STAGE_ID].csv"
                    },
                    "feature_importance": {
                        "name": "lr_feature_importance_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "interaction_params": {
                        "save_frequency": -1,
                        "write_training_prediction": True,
                        "write_validation_prediction": True,
                        "echo_training_metrics": True
                    },
                    "train_params": {
                        "global_epoch": 2,
                        "batch_size": 512,
                        "encryption": {
                            "ckks": {
                                "poly_modulus_degree": 8192,
                                "coeff_mod_bit_sizes": [
                                    60,
                                    40,
                                    40,
                                    60
                                ],
                                "global_scale_bit_size": 40
                            }
                        },
                        "optimizer": {
                            "lr": 0.01,
                            "p": 2,
                            "alpha": 1e-4
                        },
                        "metric": {
                            "decision_table": {
                                "method": "equal_frequency",
                                "bins": 10
                            },
                            "acc": {},
                            "precision": {},
                            "recall": {},
                            "f1_score": {},
                            "auc": {},
                            "ks": {}
                        },
                        "early_stopping": {
                            "key": "acc",
                            "patience": 10,
                            "delta": 0
                        },
                        "random_seed": 50
                    }
                }
            }
        ]
        
        request.dumpedTrainConfig = json.dumps(conf)
        # request.existedInputPath.append()
        response = start_client.checkTaskConfig(request)
        # print("-------")
        # print(text_format.MessageToString(response.multiStageResult))
        # print(response.message)
        # print(response.code)
        # print(response)
        m = text_format.MessageToString(response.crossStageResult)
        
        assert m.replace(' ', '').replace('\n', '') == '''
        duplicatedInputOutput {
                dumpedValue: "\\"/tmp/xfl/checkpoints/JOB_ID/NODE_ID/vertical_logitstic_regression_4.pt\\""
                positionList {
                    stageId: 4
                    pathInfo {
                    dictPath {
                        key: "model"
                    }
                    }
                }
                positionList {
                    stageId: 5
                    pathInfo {
                    dictPath {
                        key: "model"
                    }
                    }
                }
                }
                blankInputOutput {
                dumpedValue: "\\"\\""
                positionList {
                    stageId: 5
                    pathInfo {
                    dictPath {
                        key: "pretrained_model"
                    }
                    }
                }
                }
                nonexistentInput {
                dumpedValue: "\\"/tmp/xfl/dataset/testing/fintech/banking_guest_train_v01_20220216_TL.csv\\""
                positionList {
                    pathInfo {
                    dictPath {
                        key: "trainset"
                    }
                    }
                }
                }
                nonexistentInput {
                dumpedValue: "\\"/tmp/xfl/dataset/testing/fintech/banking_guest_train_v01_20220216_TL.csv\\""
                positionList {
                    stageId: 1
                    pathInfo {
                    dictPath {
                        key: "trainset"
                    }
                    }
                }
                }
                nonexistentInput {
                dumpedValue: "\\"/tmp/xfl/dataset/testing/fintech/banking_guest_train_v01_20220216_TL.csv\\""
                positionList {
                    stageId: 1
                    pathInfo {
                    dictPath {
                        key: "valset"
                    }
                    }
                }
                }
                nonexistentInput {
                dumpedValue: "\\"/tmp/xfl/checkpoints/JOB_ID/NODE_ID/woe_iv_result_3.json\\""
                positionList {
                    stageId: 3
                    pathInfo {
                    dictPath {
                        key: "iv_result"
                    }
                    }
                }
                }
                nonexistentInput {
                dumpedValue: "\\"\\""
                positionList {
                    stageId: 5
                    pathInfo {
                    dictPath {
                        key: "pretrained_model"
                    }
                    }
                }
                }
        
        '''.replace(' ', '').replace('\n', '')
