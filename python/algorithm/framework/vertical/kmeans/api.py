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

from .table_agg_otp import TableAggregatorOTPAssistTrainer, TableAggregatorOTPTrainer
from .table_agg_plain import TableAggregatorPlainAssistTrainer, TableAggregatorPlainTrainer


def _get_table_agg_inst(role: str, sec_conf: dict, *args, **kwargs) -> Union[TableAggregatorPlainTrainer,
                                                                             TableAggregatorPlainAssistTrainer]:
    """ get a proper TableAggregator instance. role: "trainer" or "scheduler"
    """

    if sec_conf is None or not sec_conf:  # sec_conf may be None or {}
        method = "plain"
        sec_conf = {
            "plain": {}
        }
    else:
        method = list(sec_conf.keys())[0]

    opt = {
        "otp": {
            "trainer": TableAggregatorOTPTrainer,
            "scheduler": TableAggregatorOTPAssistTrainer
        },
        "plain": {
            "trainer": TableAggregatorPlainTrainer,
            "scheduler": TableAggregatorPlainAssistTrainer
        }
    }

    try:
        return opt[method][role](sec_conf[method], *args, **kwargs)
    except KeyError as e:
        raise KeyError("Combination of method {} and role {} is not supported "
                       "for creating TableAggregator instance".format(method, role)) from e
    except Exception as e:
        raise e


def get_table_agg_scheduler_inst(sec_conf: dict, *args, **kwargs) -> Union[TableAggregatorPlainAssistTrainer,
                                                                           TableAggregatorOTPAssistTrainer]:
    return _get_table_agg_inst('scheduler', sec_conf, *args, **kwargs)


def get_table_agg_trainer_inst(sec_conf: dict, *args, **kwargs) -> Union[TableAggregatorPlainTrainer,
                                                                         TableAggregatorOTPTrainer]:
    return _get_table_agg_inst('trainer', sec_conf, *args, **kwargs)
