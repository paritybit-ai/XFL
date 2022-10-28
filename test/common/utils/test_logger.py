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

import os
import shutil
from common.utils.logger import add_job_log_handler, add_job_stage_log_handler, remove_log_handler, logger

LOG_PATH = "/opt/log"


def test_add_job_log_handler():
    job_handler = add_job_log_handler("unit_test")
    logger.info("add_job_log_handler")
    assert job_handler.baseFilename == "/opt/log/unit_test/xfl.log"
    assert os.path.exists("/opt/log/unit_test/xfl.log")
    with open("/opt/log/unit_test/xfl.log") as f:
        assert f.readline().split()[-1] == "add_job_log_handler"
    shutil.rmtree("/opt/log/unit_test")


def test_add_job_stage_log_handler():
    job_stage_handler = add_job_stage_log_handler("unit_test", "test_model")
    logger.info("add_job_stage_log_handler")
    assert job_stage_handler.baseFilename  == "/opt/log/unit_test/test_model.log"
    assert os.path.exists("/opt/log/unit_test/test_model.log")
    with open("/opt/log/unit_test/test_model.log") as f:
        assert f.readline().split()[-1] == "add_job_stage_log_handler"
    shutil.rmtree("/opt/log/unit_test")



def test_remove_log_handler():
    job_handler = add_job_log_handler("unit_test")
    remove_log_handler(job_handler)
    shutil.rmtree("/opt/log/unit_test")