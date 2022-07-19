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
import os
import shutil
from pathlib import Path

from common.utils.utils import save_model_config


def test_save_model_config():
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test/")
        
    p = Path("/opt/checkpoints/unit_test")
    save_model_config([{"node-1":{}},{"node-2":{}}], p)
    assert os.path.isfile("/opt/checkpoints/unit_test/model_config.json")

    save_model_config([{"node-3":{}},{"node-4":{}}], p)
    assert os.path.isfile("/opt/checkpoints/unit_test/model_config.json")
    with open("/opt/checkpoints/unit_test/model_config.json") as f:
        data = json.load(f)
    assert len(data) == 4

    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test/")
