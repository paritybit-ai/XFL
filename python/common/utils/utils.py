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
from pathlib import Path
from typing import List
import time 
from functools import wraps


def save_model_config(stage_model_config: List[dict], save_path: Path) -> None:
    """ Model config preserver.

    Args:
        stage_model_config: List[dict], Single stage model config.
        save_path: Save path.

    Returns: None

    """
    total_path = os.path.join(save_path, "model_config.json")
    if len(stage_model_config) == 0:
        raise TypeError("Length of stage_model_config should larger than 0.")
    if not os.path.exists(save_path):
        save_path.mkdir(parents=True, exist_ok=True)
    # if file not exists, create one then init first stage in it.
    if not os.path.exists(total_path):
        with open(total_path, "w") as wf1:
            json.dump(stage_model_config, fp=wf1)
    else:
        with open(total_path, "r") as f:
            org_data = json.load(f)
        org_data += stage_model_config
        with open(total_path, "w") as wf:
            json.dump(org_data, fp=wf)


def func_timer(func):
    @wraps(func)
    def with_time(*args, **kwargs):
        local_time = time.time()
        print(func.__name__ + " was called")
        f = func(*args, **kwargs)
        print(f"{func.__name__} cost {time.time()-local_time}s")
        return f
    return with_time
