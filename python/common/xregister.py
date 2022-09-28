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


from typing import Callable, Union

from common.utils.logger import logger


class XRegister():
    """ Class for register object, for example loss, metric, dataset, etc..
    """     
    @property
    def registered_object(self):
        return self.__dict__

    @classmethod
    def get_class_name(cls):
        return cls.__name__
    
    def __call__(self, name: str) -> Callable:
        if not hasattr(self, name):
            raise KeyError(f"Calling {name} not registed in {self.get_class_name()}.")
        return getattr(self, name)
        
    def register(self, target: Union[Callable, str]):
        def add_register_item(key: str, value: Callable):
            if not callable(value):
                raise TypeError(f"Register object {value} is not callable.")
            if hasattr(self, key):
                logger.warning(f"Repeated register key {key} to {self.get_class_name()}.")
            setattr(self, key, value)
            logger.info(f"Register {key} to {self.get_class_name()} successfully.")
            return target
        
        if callable(target):
            return add_register_item(target.__name__, target)
        return lambda x: add_register_item(target, x)
    
    def unregister(self, name: str):
        if hasattr(self, name):
            delattr(self, name)
            logger.info(f"Unregister {name} from {self.get_class_name()} successfully.")
        else:
            logger.warning(f"Try to unregister an non-exist key {name} from {self.get_class_name()}.")


xregister = XRegister()
