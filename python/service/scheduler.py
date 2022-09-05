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

from common.communication.gRPC.python import (commu_pb2, control_pb2,
                                              scheduler_pb2, status_pb2)
from common.storage.redis.redis_conn import RedisConn
from common.utils.logger import logger
from service.fed_config import FedConfig
from service.fed_control import get_trainer_status, trainer_control
from service.fed_job import FedJob


class SchedulerService(object):
    def __init__(self):
        pass
    
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
        config_str = json.dumps(FedConfig.trainer_config[FedJob.current_stage][request.nodeId])
        response.config = config_str
        response.jobId = FedJob.job_id
        response.code = 0
        return response

    def control(self, request, context):
        response = control_pb2.ControlResponse()
        response.code = 0

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
            response.message = "Ack"

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
            job_status = status_pb2.Status()
            response.jobId = request_job_id
            if request_job_id == FedJob.job_id and FedJob.status == status_pb2.TRAINING:
                job_status.code = status_pb2.TRAINING
                job_status.status = status_pb2.StatusEnum.Name(status_pb2.TRAINING)
                response.jobStatus.CopyFrom(job_status)
                return response

            redis_job_status = RedisConn.get("XFL_JOB_STATUS_"+str(request_job_id))
            if int(redis_job_status) == status_pb2.SUCCESSFUL:
                job_status.code = int(redis_job_status)
                job_status.status = status_pb2.StatusEnum.Name(int(redis_job_status))
            else:
                job_status.code = status_pb2.FAILED
                job_status.status = status_pb2.StatusEnum.Name(status_pb2.FAILED)
            response.jobStatus.CopyFrom(job_status)
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

    def getStage(self, request, context):
        response = scheduler_pb2.GetStageResponse()
        try:
            stage_config = FedConfig.trainer_config[FedJob.current_stage]
            if len(stage_config) < 1:
                response.code = 1
                stage_name = ""
            else:
                response.code = 0
                stage_config = list(stage_config.values())[0]
                stage_name = stage_config.get("model_info", {}).get("name", "")
        except IndexError:
            response.code = 2
            stage_name = ""
        response.stageId = FedJob.current_stage
        response.stageName = stage_name
        return response
