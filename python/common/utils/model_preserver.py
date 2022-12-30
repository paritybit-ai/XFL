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
from typing import OrderedDict

import torch

from common.utils.logger import logger


class ModelPreserver(object):
    @staticmethod
    def save(save_dir: str,
             model_name: str,
             state_dict: OrderedDict,
             epoch: int = None,
             final: bool = False,
             suggest_threshold: float = None
             ):

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        model_info = {"state_dict": state_dict}
        if suggest_threshold:
            model_info["suggest_threshold"] = suggest_threshold

        model_name_list = model_name.split(".")
        name_prefix, name_postfix = ".".join(model_name_list[:-1]), model_name_list[-1]

        if not final and epoch:
            model_name = name_prefix + "_epoch_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix

        model_path = os.path.join(save_dir, model_name)
        torch.save(model_info, model_path)
        logger.info("model saved as: {}.".format(model_path))
        return

    @staticmethod
    def load(model_path: str):
        return torch.load(model_path)
