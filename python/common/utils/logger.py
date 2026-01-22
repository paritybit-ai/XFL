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


import logging.config
import os
from logging import FileHandler, LogRecord

LOG_PATH = os.environ.get("XFL_LOG_PATH", os.path.join(os.getcwd(), "log"))

class ColorFormatter(logging.Formatter):
    log_colors = {
        'CRITICAL': '\033[0;31m',
        'ERROR': '\033[0;33m',
        'WARNING': '\033[0;35m',
        'INFO': '\033[0;32m',
        'DEBUG': '\033[0;00m',
    }

    def format(self, record: LogRecord) -> str:
        s = super().format(record)

        level_name = record.levelname
        if level_name in self.log_colors:
            return self.log_colors[level_name] + s + '\033[0m'
        return s

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

# format
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
color_formatter = ColorFormatter("%(asctime)s %(levelname)s: %(message)s")

# console output
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(color_formatter)
logger.addHandler(streamHandler)


def get_node_log_path(job_id: str, node_ids: list[str]):
    log_path = {}
    for node_id in node_ids:
        path = "{}/{}/{}/xfl.log".format(LOG_PATH, job_id, node_id)
        log_path[node_id] = path
    return log_path


def get_stage_node_log_path(job_id: str, train_conf: dict):
    stages_log_path = {}
    for stage_id, node_conf in train_conf.items():
        stages_log_path[stage_id] = {}
        for node_id, conf in node_conf.items():
            model_name = conf.get('model_info', {}).get('name', '')
            if model_name == '':
                continue
            path = "{}/{}/{}/stage{}_{}.log".format(LOG_PATH, job_id, node_id, stage_id, model_name)
            stages_log_path[stage_id][node_id] = path
    return stages_log_path


def add_job_log_handler(job_id: str, node_id: str) -> object:
    if not os.path.exists("{}/{}/{}".format(LOG_PATH, job_id, node_id)):
        os.makedirs("{}/{}/{}".format(LOG_PATH, job_id, node_id))
    job_handler = FileHandler("{}/{}/{}/xfl.log".format(LOG_PATH, job_id, node_id))
    job_handler.setFormatter(formatter)
    logger.addHandler(job_handler)
    return job_handler


def add_job_stage_log_handler(job_id: str, node_id: str, stage_id: int, model_name: str) -> object:
    if model_name == '':
        return None
    if not os.path.exists("{}/{}/{}".format(LOG_PATH, job_id, node_id)):
        os.makedirs("{}/{}/{}".format(LOG_PATH, job_id, node_id))
    stage_handler = FileHandler("{}/{}/{}/stage{}_{}.log".format(LOG_PATH, job_id, node_id, stage_id, model_name))
    stage_handler.setFormatter(formatter)
    logger.addHandler(stage_handler)
    return stage_handler


def remove_log_handler(handler):
    logger.removeHandler(handler)
