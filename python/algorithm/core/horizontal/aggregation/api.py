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


from typing import Union

from .aggregation_otp import AggregationOTPRoot, AggregationOTPLeaf
from .aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from service.fed_config import FedConfig


def _get_aggregation_inst(role: str, sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationPlainRoot]:
# def _get_aggregation_inst(role: str, sec_conf: dict, root_id: str, leaf_ids: list[str]) -> Union[AggregationPlainLeaf, AggregationPlainRoot]:
    """ get a proper FedAvg instance. role: "label_trainer" or "assist_trainer"
    """
    if not sec_conf or len(leaf_ids) == 1 or len(FedConfig.get_label_trainer() + FedConfig.get_trainer()) == 1:
        method = "plain"
        sec_conf = {}
    else:
        method = list(sec_conf.keys())[0]
        sec_conf = sec_conf[method]

    opt = {
        "otp": {
            "leaf": AggregationOTPLeaf,
            "root": AggregationOTPRoot
        }, 
        "plain": {
            "leaf": AggregationPlainLeaf,
            "root": AggregationPlainRoot
        }
    }

    try:
        return opt[method][role](sec_conf, root_id, leaf_ids)
    except KeyError as e:
        raise KeyError("Combination of method {} and role {} is not supported for creating FedAvg instance".format(method, role)) from e
    except Exception as e:
        raise e


def get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]:
    return _get_aggregation_inst('root', sec_conf, root_id, leaf_ids)


def get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]:
    return _get_aggregation_inst('leaf', sec_conf, root_id, leaf_ids)


