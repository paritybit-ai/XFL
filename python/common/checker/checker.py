import traceback

import numpy as np

from .qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from .x_types import String, Bool, Integer, Float, Any, All


class Checked():
    def __init__(self, value, is_match: bool, reason: str = ''):
        self.value = value
        self.is_match = is_match
        self.reason = reason
        
    def result(self):
        is_match = "Match" if self.is_match else "Not Match"

        if isinstance(self.value, dict):
            res = {
                "__match__": ':'.join([is_match, self.reason])
            }
            res.update(
                {
                    k.result() if isinstance(k, Checked) else k: v.result() if isinstance(v, Checked) else v for k, v in self.value.items()
                }
            )
        elif isinstance(self.value, list):
            res = ['__match__: ' + '-'.join([is_match, self.reason])] + [v.result() if isinstance(v, Checked) else v for v in self.value]
        elif isinstance(self.value, Checked): 
            res = self.value.result()
        else:
            res = '-'.join(['(' + str(self.value) + ')', is_match, self.reason])
        return res
    
    
def is_valid_match_num(item_to_match, num_matched):
    reason = ''
    if isinstance(item_to_match, OneOf):
        is_valid = True if num_matched == 1 else False
        if not is_valid:
            reason = f"{item_to_match.__name__}: matched {num_matched}, expect 1"
    elif isinstance(item_to_match, SomeOf):
        is_valid = True if 0 < num_matched <= len(item_to_match.candidates) else False
        if not is_valid:
            reason = f"{item_to_match.__name__}: matched {num_matched}, expect > 0 and <= {len(item_to_match.candidates)}"
    elif isinstance(item_to_match, RepeatableSomeOf):
        is_valid = True if num_matched > 0 else False
        if not is_valid:
            reason = f"{item_to_match.__name__}: matched {num_matched}, expect > 0"
    elif isinstance(item_to_match, Required):
        is_valid = True if num_matched == len(item_to_match.candidates) else False
        if not is_valid:
            reason = f"{item_to_match.__name__}: matched {num_matched}, expect {len(item_to_match.candidates)}"
    elif isinstance(item_to_match, (Optional, Any)):
        is_valid = True
    else:
        is_valid = True if num_matched == 1 else False
        if not is_valid:
            if isinstance(item_to_match, (String, Bool, Integer, Float)):
                reason = f"no match for {item_to_match.__name__}"
            else:
                reason = f"no match for {item_to_match}"
    return is_valid, reason


def find_key_matched(key, dst_keys):
    # 主要是为了处理dict规则中有Any, String等通用key的情况
    for k in dst_keys:
        if check(key, k).is_match:
            return k
    return None
    

def cal_num_valid(checked):
    if not isinstance(checked, Checked):
        return 0, 0
    
    if isinstance(checked.value, Checked):
        valid, total = cal_num_valid(checked.value)
        valid += int(checked.is_match)
        total += 1
    elif isinstance(checked.value, dict):
        valid, total = int(checked.is_match), 1
        for k, v in checked.value.items():
            valid_1, total_1 = cal_num_valid(k)
            valid_2, total_2 = cal_num_valid(v)
            valid += valid_1 + valid_2
            total += total_1 + total_2
    elif isinstance(checked.value, list):
        valid, total = int(checked.is_match), 1
        for v in checked.value:
            valid_2, total_2 = cal_num_valid(v)
            valid += valid_2
            total += total_2
    else:
        valid, total = int(checked.is_match), 1
        
    return valid, total


def check(config, rule, ori_config=None) -> Checked:
    if ori_config is None:
        ori_config = config
        
    def _check_rules(rules, config, ori_config):
        for rule, desp in rules:
            try:
                num_vars = rule.__code__.co_argcount
                if num_vars == 1:
                    is_valid = rule(config)
                else:
                    is_valid = rule(config, ori_config)
                
                if not is_valid:
                    return False, desp
            except Exception:
                traceback.print_exc()
                return False, "Cannot apply rule"
        return True, 'Additional rules passed'
    
    if isinstance(rule, String):
        flag = isinstance(config, str)
        flag2, reason = _check_rules(rule.rules, config, ori_config)
        if not flag or (flag and flag2):
            return Checked(config, flag, rule.__name__)
        else:
            return Checked(config, flag2, reason)
    if isinstance(rule, Bool):
        flag = isinstance(config, bool)
        flag2, reason = _check_rules(rule.rules, config, ori_config)
        if not flag or (flag and flag2):
            return Checked(config, flag, rule.__name__)
        else:
            return Checked(config, flag2, reason)
    if isinstance(rule, Integer):
        flag = isinstance(config, int)
        flag2, reason = _check_rules(rule.rules, config, ori_config)
        if not flag or (flag and flag2):
            return Checked(config, flag, rule.__name__)
        else:
            return Checked(config, flag2, reason)
    elif isinstance(rule, Float):
        flag = isinstance(config, float) or isinstance(config, int)
        flag2, reason = _check_rules(rule.rules, config, ori_config)
        if not flag or (flag and flag2):
            return Checked(config, flag, rule.__name__)
        else:
            return Checked(config, flag2, reason)
    elif isinstance(rule, (OneOf, Required, Optional, SomeOf, RepeatableSomeOf)):
        # config is alwary one element
        if isinstance(rule, Optional):
            res = [check(config, v, ori_config) for v in rule.candidates + (None,)]
        else:
            res = [check(config, v, ori_config) for v in rule.candidates]
        
        is_match = [i.is_match for i in res]
        num_valid = sum(is_match)
        
        if isinstance(rule, (OneOf, SomeOf, RepeatableSomeOf)):
            flag = True if num_valid == 1 else False
        elif isinstance(rule, Required):
            # Normally, Required only act on dict keys
            flag = True if num_valid == 1 else False
        else:
            flag = True if config is None or num_valid == 1 else False
            
        flag2, reason = _check_rules(rule.rules, config, ori_config)

        if flag:
            if flag2:
                pos = is_match.index(True)
                return Checked(res[pos], flag, reason=rule.__name__)
            else:
                return Checked(config, flag2, reason=reason)
        else:
            return Checked(config, flag, reason=rule.__name__)
    elif isinstance(rule, dict):
        if not isinstance(config, dict):
            return Checked(config, False, f"Type {type(config)} not match dict")
        
        if rule.get("__rule__") is None:
            rule["__rule__"] = list(rule.keys())
        
        if not isinstance(rule.get("__rule__"), list):
            rule["__rule__"] = [rule["__rule__"]]

        checked_matrix = np.array([
            [check(k, r, ori_config) for r in rule["__rule__"]] for k in config
        ])
        
        row_size = len(checked_matrix)
        
        if row_size > 0:
            col_size = len(checked_matrix[0])
            
            is_match_matrix = np.zeros_like(checked_matrix)
            
            for i in range(row_size):
                for j in range(col_size):
                    is_match_matrix[i][j] = checked_matrix[i, j].is_match
            
            num_match_list = np.sum(is_match_matrix, axis=0)
        else:
            num_match_list = [0 for i in range(len(rule["__rule__"]))]
            
        is_valid_list = [is_valid_match_num(rule["__rule__"][i], num_match_list[i]) for i in range(len(rule["__rule__"]))]
            
        is_match = True
        reason = []
        for is_valid, r in is_valid_list:
            if not is_valid:
                is_match = False
            if r:
                reason.append(r)
                
        for i in range(row_size):
            if np.sum(is_match_matrix[i]) == 0:
                is_match = False
                reason.append(f"{list(config.keys())[i]} match no rules")
                
        reason = ','.join(reason)

        result = {}
        for i, k in enumerate(list(config.keys())):
            if np.sum(is_match_matrix[i]) == 0:
                result[Checked(k, False, '')] = Checked(config[k], False, 'match no rules')
            else:
                for j, flag in enumerate(is_match_matrix[i]):
                    if flag:
                        result[checked_matrix[i][j]] = check(config[k], rule[find_key_matched(k, list(rule.keys()))], ori_config)
                        break
        
        return Checked(result, is_match, reason)
    
    elif isinstance(rule, list):
        if not isinstance(config, list):
            return Checked(config, False, f"Type {type(config)} not match list")
        
        # SomeOf和RepeatableSomeOf在这里没有什么区别
        if len(rule) == 1 and isinstance(rule[0], (OneOf, SomeOf, RepeatableSomeOf, Required, Optional, Any, All)):
            if isinstance(rule[0], Any):
                if len(config) != 1:
                    return Checked(config, False, f"List length {len(config)} != 1")
                
                res = check(config[0], rule[0], ori_config)
                return Checked([res], True, 'list')
            
            if isinstance(rule[0], All):
                return Checked(config, True, All.__name__)
            
            if isinstance(rule[0], Optional):
                if len(rule[0].candidates) != 1:
                    raise ValueError(f"Optional rule {rule} may not be well defined.")
                    
                if len(config) == 0:
                    return Checked(config, True, Optional.__name__ + "_√")
                else:
                    rule_copy = [rule[0].candidates[0]]
                    res = check(config, rule_copy, ori_config)
                    if res.is_match:
                        res.reason = Optional.__name__ + "_√"
                    else:
                        res.reason = Optional.__name__ + "_×" + "," + res.reason
                    return res
                
            checked_list = [check(v, rule[0], ori_config) for v in config]
            is_valid_list = [v.is_match for v in checked_list]
            
            is_match, r = is_valid_match_num(rule[0], sum(is_valid_list))
            
            reason = []
            if r:
                reason.append(r)
                
            if isinstance(rule[0], SomeOf):
                valid_config = []
                for i, v in enumerate(is_valid_list):
                    if v:
                        valid_config.append(config[i])
                
                if len(set(valid_config)) != len(valid_config):
                    is_match = False
                    reason.append("Repeated items for SomeOf")
                
            for i, v in enumerate(is_valid_list):
                if not v:
                    is_match = False
                    reason.append(f"{config[i]} match nothing")
            
            if is_match:
                reason.insert(0, rule[0].__name__ + "_√")
            else:
                reason.insert(0, rule[0].__name__ + "_×")
                
            reason = ','.join(reason)
            
            return Checked(checked_list, is_match, reason)

        else:
            if len(config) != len(rule):
                return Checked(config, False, f"List length {len(config)} != {len(rule)}")
            
            res = [check(config[i], rule[i], ori_config) for i in range(len(rule))]
            is_match = [v.is_match for v in res]
            num_total = len(is_match)
            num_valid = sum(is_match)
            flag = (num_valid == num_total)
            return Checked(res, flag, f"{num_valid}/{num_total}")
        
    else:
        if config == rule:
            if isinstance(rule, (Any, All)):
                is_match, reason = _check_rules(rule.rules, config, ori_config)
                if not is_match:
                    return Checked(config, False, reason)
                else:
                    return Checked(config, True, rule.__name__)
            else:
                return Checked(config, True, str(rule))
        else:
            if rule is None:
                return Checked(config, False, "no rule")
            else:
                return Checked(config, False, "not equal")