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


from .checker import check, Checked


# For sync in algorithms
def get_matched_config(config, rule):
    r = check(config, rule)
    
    def get_matched(checked):
        if isinstance(checked, Checked):
            if isinstance(checked.value, dict):
                tmp = {}
                for k, v in checked.value.items():
                    if hasattr(k, 'is_match'):
                        if k.is_match:
                            tmp.update({k.value: get_matched(v)})
                    else:
                        tmp.update({k: get_matched(v)})
                return tmp
            else:
                if checked.is_match:
                    return checked.value
                else:
                    return None
        else:
            return checked
    
    return get_matched(r)
    
