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


from importlib import import_module

from common.xregister import xregister


def get_operator(name: str, role: str) -> object:
    """ Get operator by name and role.

    Args:
        name (str): operator name.
        role (str): assist_trainer, trainer or label_trainer.

    Returns:
        Operator find by name and role.
    """
    if role not in ["assist_trainer", "trainer", "label_trainer"]:
        raise ValueError(f"Identity {role} is not valid, need to be assist_trainer, trainer or label_trainer.")
    
    fed_type = name.split("_")[0]
    operator_dir = "_".join(name.split("_")[1:])
    
    if fed_type not in ["horizontal", "vertical", "local"]:
        raise ValueError(f"Prefix of operator name {name} is not valid, need to be horizontal, vertical or local.")
        
    class_name = [fed_type] + operator_dir.split("_") + role.split("_")
    class_name = ' '.join(class_name).title().replace(' ', '')
    module_path = '.'.join(["algorithm", "framework", fed_type, operator_dir, role])

    try:
        module = import_module(module_path)
        operator = getattr(module, class_name)
    except ModuleNotFoundError:
        if class_name in xregister.registered_object:
            operator = xregister(class_name)
        else:
            raise ValueError(f"Operator name: {name}, role: {role} is not defined.")
    return operator
