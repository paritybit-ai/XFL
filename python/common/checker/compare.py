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


from .checker import check, cal_num_valid


def compare(config, rule):
    r = check(config, rule)
    rule_passed, rule_checked = cal_num_valid(r)
    # num_valid, num_total = cal_num_valid(r)
    
    # result = r.result()
    result = r.breif_result()
    itemized_result = r.get_unmatch_position()
    # print(position, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # if isinstance(result, dict):
    #     result['rule_passed'] = num_valid
    #     result['rule_checked'] = num_total
    #     # result["__summary__"] = f"{num_valid}/{num_total}"
    # elif isinstance(result, list):
    #     # result.insert(0, f"__summary__: ({num_valid}/{num_total})")
    #     result.insert(0, f"__rule_passed: {num_valid}")
    #     result.insert(1, f"__rule_checked: {num_total}")
        
    return result, itemized_result, rule_passed, rule_checked
    
    
    
    

    
    