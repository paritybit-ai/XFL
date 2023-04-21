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
import json
from typing import Optional
from pathlib import Path
import torch

from common.utils.logger import logger


class ModelIO:
    @staticmethod
    def _gen_model_path(save_dir: str, model_name: str, epoch: Optional[int] = None):
        split_name = model_name.split(".")
        if epoch is None:
            model_name = '.'.join(split_name[:-1]) + '.' + split_name[-1]
        else:
            model_name = '.'.join(split_name[:-1]) + f'_epoch_{epoch}.' + split_name[-1]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_path = Path(save_dir, model_name)
        return model_path

    @staticmethod
    def save_torch_model(state_dict, 
                         save_dir: str, 
                         model_name: str, 
                         meta_dict: dict = {}, 
                         epoch: Optional[int] = None, 
                         version: str = '1.4.0'):
        model_dict = {}
        model_dict.update(meta_dict)
        model_dict = {"state_dict": state_dict, "version": version}
        model_path = ModelIO._gen_model_path(save_dir, model_name, epoch)
        torch.save(model_dict, model_path)
        logger.info("Model saved as: {}".format(model_path))
        
    @staticmethod
    def load_torch_model(model_path: str):
        model_dict = torch.load(model_path)
        logger.info("Model loaded from: {}".format(model_path))
        return model_dict
    
    @staticmethod
    def save_torch_onnx(model, input_dim: tuple, save_dir: str, model_name: str, epoch: Optional[int] = None):
        dummy_input = torch.randn(1, *input_dim)
        model_path = ModelIO._gen_model_path(save_dir, model_name, epoch)

        torch.onnx.export(model,
                          dummy_input,
                          model_path,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        logger.info("Model saved as: {}".format(model_path))
        
    @staticmethod
    def save_json_model(model_dict: dict,
                        save_dir: str,
                        model_name: str,
                        meta_dict: dict = {}, 
                        epoch: Optional[int] = None,
                        version: str = '1.4.0'):
        new_model_dict = {}
        new_model_dict.update(meta_dict)
        new_model_dict.update(model_dict)
        new_model_dict["version"] = version
        model_path = ModelIO._gen_model_path(save_dir, model_name, epoch)
        fp = open(model_path, 'w')
        json.dump(new_model_dict, fp)
        logger.info("Model saved as: {}".format(model_path))
        
    @staticmethod
    def load_json_model(model_path: str):
        with open(model_path, 'r') as fp:
            model_dict = json.load(fp)
        logger.info("Model loaded from: {}".format(model_path))
        return model_dict
    
    @staticmethod
    def save_json_proto(model_dict: dict,
                        save_dir: str,
                        model_name: str,
                        meta_dict: dict = {}, 
                        epoch: Optional[int] = None,
                        version: str = '1.4.0'):
        pass
    
    @staticmethod
    def load_json_proto(model_path: str):
        pass



