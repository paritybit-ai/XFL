import pytest

from common.checker.checker import check, cal_num_valid, find_key_matched
from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


def test_dict():
    config = {"a": 1}
    rule = {"a": {}}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    
def test_list():
    config = {"a": {}}
    rule = {"a": []}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [5]}
    rule = {
        "a": [Optional(Integer())]
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": []}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    # Optional
    
    rule = {"a": [Optional(3, 5)]}
    with pytest.raises(ValueError):
        r = check(config, rule)
        
    config = {"a": [7, 3]}
    rule = {"a": [Optional(SomeOf(3, 5, 7))]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": [7, 8]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    # Any
    rule = {"a": [Any()]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [9]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": [[7, 8]]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    rule = {"a": [All()]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": [1, 1, 2]}
    rule = {"a": [SomeOf(1, 2)]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    rule = {"a": [OneOf(1, 3), 2]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    
def test_rest():
    key = "a"
    dst_keys = ["b", "c"]
    assert find_key_matched(key, dst_keys) is None
    
    config = {"a": [1]}
    rule = {Integer(): [Integer()], "b": [1]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": {
        "b": [1, 2, 3]
    }}
    rule = {String(): {
        "b": [RepeatableSomeOf(Integer())]
    }}
    r = check(config, rule)
    r.result()
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total

