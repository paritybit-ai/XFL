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


import copy
import json
import time
import traceback

import pickle
from google.protobuf import json_format

from common.communication.gRPC.python import (commu_pb2, control_pb2,
                                              scheduler_pb2, status_pb2, checker_pb2)
from common.storage.redis.redis_conn import RedisConn
from common.utils.config_parser import replace_variable
from common.utils.config_checker import check_multi_stage_train_conf, check_cross_stage_input_output
from common.utils.logger import logger, get_node_log_path, get_stage_node_log_path
from service.fed_config import FedConfig
from service.fed_control import get_trainer_status, trainer_control
from service.fed_job import FedJob
from service.fed_node import FedNode
from tqdm import trange


class SchedulerService(object):
    def __init__(self, is_bar=False):
        self.is_bar = is_bar
        self.progress_bar = None
    
    def post(self, request, context):
        request_key = ''
        request_value = bytearray()
        
        for i, r in enumerate(request):
            request_value += r.value
            if i == 0:
                request_key = r.key
                # request_info = r.key.split("~")
                # name = request_info[1]
                # start_end_id = request_info[-1]
                # logger.info(f"Start receiving the data of channel {name} from {start_end_id} ...")
            
        RedisConn.put(request_key, bytes(request_value))

        response = commu_pb2.PostResponse()
        response.code = 0
        # logger.info(f"Successfully received the data of channel {name} from {start_end_id}")
        return response

    def getConfig(self, request, context):
        response = scheduler_pb2.GetConfigResponse()
        config = copy.deepcopy(FedConfig.trainer_config[FedJob.current_stage][request.nodeId])
        config = replace_variable(config, stage_id=FedJob.current_stage, job_id=FedJob.job_id, node_id=request.nodeId)
        response.config = json.dumps(config)
        response.jobId = FedJob.job_id
        response.code = 0
        return response

    def control(self, request, context):
        response = control_pb2.ControlResponse()
        response.code = 0
        # response.logPath = json.dumps({})
        # response.nodeStageLogPath = json.dumps({})

        if request.control == control_pb2.STOP:
            FedJob.status = status_pb2.FAILED
            response.message += f"Stop Scheduler Successful.\n"
            trainer_resp = trainer_control(control_pb2.STOP)
            response.code = trainer_resp.code
            response.message += trainer_resp.message
            logger.info("Model training is stopped.")
        elif request.control == control_pb2.START:
            if FedJob.status not in (status_pb2.IDLE, status_pb2.FAILED, status_pb2.SUCCESSFUL):
                response.code = 1
                response.message = "Scheduler not ready."
                response.jobId = int(FedJob.job_id)
                return response
            else:
                resp = get_trainer_status()
                for i in resp.keys():
                    if resp[i].code not in (status_pb2.IDLE, status_pb2.FAILED, status_pb2.SUCCESSFUL):
                        response.code = 1
                        response.message = f"Trainer {i} not ready.."
                        response.jobId = int(FedJob.job_id)
                        return response

            FedJob.job_id = int(RedisConn.incr("XFL_JOB_ID"))
            RedisConn.set("XFL_JOB_STATUS_" + str(FedJob.job_id), status_pb2.TRAINING)
            FedJob.status = status_pb2.TRAINING
            
            start = time.time()
            while time.time() - start < 5:
                if FedConfig.converted_trainer_config != {}:
                    response.message = 'Ack'
                    response.dumpedTrainConfig = json.dumps(FedConfig.converted_trainer_config)
                    
                    raw_node_log_path = get_node_log_path(job_id=FedJob.job_id, node_ids=list(FedNode.trainers.keys()) + ['scheduler'])
                    raw_stage_node_log_path = get_stage_node_log_path(job_id=FedJob.job_id, train_conf=FedConfig.converted_trainer_config)
                    
                    for node_id in raw_node_log_path:
                        node_log_path = scheduler_pb2.control__pb2.NodeLogPath()
                        node_log_path.nodeId = node_id
                        node_log_path.logPath = raw_node_log_path[node_id]
                        response.nodeLogPath.append(node_log_path)
                        
                    for stage_id in raw_stage_node_log_path:
                        for node_id in raw_stage_node_log_path[stage_id]:
                            stage_node_log_path = scheduler_pb2.control__pb2.StageNodeLogPath()
                            stage_node_log_path.stageId = int(stage_id)
                            stage_node_log_path.nodeId = node_id
                            stage_node_log_path.logPath = raw_stage_node_log_path[stage_id][node_id]
                            response.stageNodeLogPath.append(stage_node_log_path)
                    
                    FedConfig.converted_trainer_config = {}
                    break
                time.sleep(0.1)

        response.jobId = int(FedJob.job_id)
        return response

    def status(self, request, context):
        response = status_pb2.StatusResponse()
        request_job_id = int(request.jobId)
        
        if request_job_id == 0:
            # return node status
            node_status = status_pb2.Status()
            response.jobId = FedJob.job_id
            node_status.code = FedJob.status
            node_status.status = status_pb2.StatusEnum.Name(FedJob.status)
            response.schedulerStatus.CopyFrom(node_status)
            resp = get_trainer_status()
            for t in resp.keys():
                response.trainerStatus[t].CopyFrom(resp[t])
        elif request_job_id <= FedJob.job_id:
            # return job status
            start_time = RedisConn.get("XFL_JOB_START_TIME_"+str(request_job_id)) or 0
            job_status = status_pb2.Status()
            response.jobId = request_job_id
            if request_job_id == FedJob.job_id and FedJob.status == status_pb2.TRAINING:
                job_status.code = status_pb2.TRAINING
                job_status.status = status_pb2.StatusEnum.Name(status_pb2.TRAINING)
                response.jobStatus.CopyFrom(job_status)
                response.startTime = int(start_time)
                return response

            redis_job_status = RedisConn.get("XFL_JOB_STATUS_"+str(request_job_id))
            if int(redis_job_status) == status_pb2.SUCCESSFUL:
                job_status.code = int(redis_job_status)
                job_status.status = status_pb2.StatusEnum.Name(int(redis_job_status))
            else:
                job_status.code = status_pb2.FAILED
                job_status.status = status_pb2.StatusEnum.Name(status_pb2.FAILED)
            response.jobStatus.CopyFrom(job_status)
            
            end_time = RedisConn.get("XFL_JOB_END_TIME_"+str(request_job_id)) or 0
            response.startTime = int(start_time)
            response.endTime = int(end_time)
        return response
    
    def checkTaskConfig(self, request, context):
        response = checker_pb2.CheckTaskConfigResponse()
        try:
            first_message = True
            response.message = ''
            configs = json.loads(request.dumpedTrainConfig)
            result_multi_stage = check_multi_stage_train_conf(configs)
            for stage_id, stage_result in enumerate(result_multi_stage["result"]):
                stage_result = checker_pb2.StageResult()
                stage_result.stageId = stage_id
                stage_result.dumpedCheckedConfig = json.dumps(result_multi_stage["result"])
                
                for itemized_info in result_multi_stage["itemized_result"][stage_id]:
                    item_info = checker_pb2.ItemInfo()
                    
                    num_path = 0
                    for info in itemized_info[:-1]:
                        path_info = checker_pb2.PathInfo()
                        if info['type'] == 'dict':
                            path_info.dictPath.key = info['key']
                            if first_message:
                                response.message += configs[stage_id].get('model_info', {}).get('name')
                                response.message += '-' + str(info['key'])
                        else:
                            path_info.listPath.index = info['index']
                            if first_message:
                                response.message += '-' + str(info['index'])
                        num_path += 1
                        
                        item_info.pathInfo.append(path_info)
                    
                    if first_message:
                        if num_path < 2:
                            response.message = ''
                        else:
                            response.message = '-'.join(response.message.split('-')[:1] + response.message.split('-')[-1:])
                            if len(itemized_info) > 0:
                                # response.message += ':' + itemized_info[-1]
                                response.message += ': format error'
                                first_message = False
                                response.code = 1
                            else:
                                response.message = ''

                    if len(itemized_info) > 0:
                        item_info.notes = itemized_info[-1]
                    
                    stage_result.unmatchedItems.append(item_info)

                    # path_info = []
                    # for info in itemized_info[:-1]:
                    #     if info['type'] == 'dict':
                    #         path_info.append(str(info['key']))
                    #     else:
                    #         path_info.append(str(info['index']))
                    # path_info = '-'.join(path_info)
                    # path_info += ":" + itemized_info[-1]
                    
                    # if itemized_info[-1]:
                    #     if path_info not in stage_result.unmatchedItems:
                    #         stage_result.unmatchedItems.append(path_info)

                stage_result.passedRules = result_multi_stage["summary"][stage_id][0]
                stage_result.checkedRules = result_multi_stage["summary"][stage_id][1]
                stage_result.code = 0
                
                response.multiStageResult.stageResultList.append(stage_result)
            response.multiStageResult.code = 0
        except Exception:
            logger.error(traceback.format_exc())
            response.multiStageResult.code = 1
            
        try:
            result_cross_stage = check_cross_stage_input_output(json.loads(request.dumpedTrainConfig), ignore_list=request.existedInputPath)
            for item in result_cross_stage['duplicated']:
                item_info = checker_pb2.CrossStageItemInfo()
                item_info.dumpedValue = json.dumps(item['value'])
                for position in item['position']:
                    position_info = checker_pb2.CrossStagePositionInfo()
                    position_info.stageId = position['stage']
                    for key in position['key_chain']:
                        path_info = checker_pb2.PathInfo()
                        path_info.dictPath.key = key
                    position_info.pathInfo.append(path_info)
                    item_info.positionList.append(position_info)
                response.crossStageResult.duplicatedInputOutput.append(item_info)
            
            for item in result_cross_stage['blank']:
                item_info = checker_pb2.CrossStageItemInfo()
                item_info.dumpedValue = json.dumps(item['value'])
                for position in item['position']:
                    position_info = checker_pb2.CrossStagePositionInfo()
                    position_info.stageId = position['stage']
                    
                    for key in position['key_chain']:
                        path_info = checker_pb2.PathInfo()
                        path_info.dictPath.key = key
                    position_info.pathInfo.append(path_info)
                    item_info.positionList.append(position_info)
                response.crossStageResult.blankInputOutput.append(item_info)
                
            for item in result_cross_stage['nonexistent']:
                item_info = checker_pb2.CrossStageItemInfo()
                item_info.dumpedValue = json.dumps(item['value'])
                for position in item['position']:
                    position_info = checker_pb2.CrossStagePositionInfo()
                    position_info.stageId = position['stage']
                    
                    for key in position['key_chain']:
                        path_info = checker_pb2.PathInfo()
                        path_info.dictPath.key = key
                    position_info.pathInfo.append(path_info)
                    item_info.positionList.append(position_info)
                response.crossStageResult.nonexistentInput.append(item_info)
            response.crossStageResult.code = 0
        except Exception:
            logger.error(traceback.format_exc())
            response.crossStageResult.code = 1
            
        return response

    def getAlgorithmList(self, request, context):
        response = scheduler_pb2.GetAlgorithmListResponse()
        response.algorithmList.extend(FedConfig.algorithm_list)
        for i in FedConfig.default_config_map.keys():
            dc = scheduler_pb2.DefaultConfig()
            for j in FedConfig.default_config_map[i].keys():
                dc.config[j] = json.dumps(FedConfig.default_config_map[i][j])
            response.defaultConfigMap[i].CopyFrom(dc)
        return response
    
    def recProgress(self, request, context):
        if self.is_bar:
            if (self.progress_bar is None and FedJob.progress[FedJob.current_stage] == 0) or \
                (self.progress_bar and self.progress_bar.desc != f"Stage {FedJob.current_stage}"):
                    self.progress_bar = trange(FedJob.max_progress, desc=f"Stage {FedJob.current_stage}")
                    
        response = scheduler_pb2.RecProgressResponse()
        response.code = 1
        if request.progress > FedJob.progress[FedJob.current_stage]:

            if self.is_bar:
                self.progress_bar.update(request.progress-FedJob.progress[FedJob.current_stage])
                if request.progress == FedJob.max_progress:
                    self.progress_bar.close()
                    self.progress_bar = None

            FedJob.progress[FedJob.current_stage] = request.progress
            response.code = 0
            
        stage_response = scheduler_pb2.GetStageResponse()
        try:
            stage_config = FedConfig.trainer_config[FedJob.current_stage]
            if len(stage_config) < 1:
                stage_response.code = 1
                stage_name = ""
            else:
                response.code = 0
                stage_config = list(stage_config.values())[0]
                stage_name = stage_config.get("model_info", {}).get("name", "")
        except IndexError:
            stage_response.code = 2
            stage_name = ""
        stage_response.currentStageId = FedJob.current_stage
        stage_response.totalStageNum = FedJob.total_stage_num
        stage_response.currentStageName = stage_name
        
        bar_response = scheduler_pb2.ProgressBar()
        for stage, progress in enumerate(FedJob.progress):
            bar_response.stageId = stage
            bar_response.stageProgress = progress
            stage_response.progressBar.append(bar_response)
            
        RedisConn.set("XFL_JOB_STAGE_" + str(FedJob.job_id), json_format.MessageToJson(stage_response))
        return response

    def getStage(self, request, context):
        job_id = request.jobId
        key = "XFL_JOB_STAGE_" + str(job_id)
        response = RedisConn.get(key)
        if not response:
            response = scheduler_pb2.GetStageResponse()
            response.code = 3
        else:
            response = json_format.Parse(response, scheduler_pb2.GetStageResponse())
            response.isRunning = True if FedJob.job_id == job_id and FedJob.status == status_pb2.TRAINING else False
        return response
