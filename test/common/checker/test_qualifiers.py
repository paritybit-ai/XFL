import pytest

from common.checker.checker import check, cal_num_valid
from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


def test_OneOf():
    config = {"a": 1}
    rule = {"a": OneOf(1, 2).set_default(1)}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["a"].default == 1
    
    rule = {
        "__rule__": OneOf("a", "b").set_default_index(0),
        "a": Integer()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["__rule__"][0].default == "a"
    
    config = {"c": 3.4}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [1, 2, 3]}
    rule = {
        "a": [OneOf(1, 2, 3), OneOf(1, 2, 3), OneOf(1, 2, 3).add_rule(lambda x: x < config["a"][1])]
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [1, 2, 3]}
    rule = {
        "a": [OneOf(1, 2, 3), OneOf(1, 2, 3), OneOf(1, 2, 3).add_rule(lambda x, y: x < y["a"][1])]
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [1, 2, 3]}
    rule = {
        "a": [OneOf(1, 2, 3), OneOf(1, 2, 3), OneOf(1, 2, 3).add_rule(lambda x, y: x < y["b"][1])]
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total


def test_SomeOf():
    config = {"a": [1, 2]}
    rule = {"a": [SomeOf(1, 2, 3).set_default([2, 3])]}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["a"][0].default == [2, 3]
    
    rule = {"a": [SomeOf(3, 4)]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": 3, "b": 5}
    rule = {
        "__rule__": SomeOf("a", "b", "c").set_default_indices(1, 2),
        "a": Integer(),
        "b": Float()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["__rule__"][0].default == ["b", "c"]
    
    rule = {
        "__rule__": SomeOf("b", "c"),
        "b": Integer(),
        "c": Float()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    
def test_RepeatableSomeOf():
    config = {"a": [1, 2, 2]}
    rule = {"a": [RepeatableSomeOf(1, 2, 3).set_default([2, 2, 3])]}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["a"][0].default == [2, 2, 3]
    
    rule = {"a": [RepeatableSomeOf(2, 3).set_default_indices(1, 0, 1)]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    assert rule["a"][0].default == [3, 2, 3]
    
    rule = {"a": [RepeatableSomeOf(3, 4)]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total

    

def test_Required():
    config = {"a": 3, "b": 4}
    rule = {
        "__rule__": Required("a", "b"),
        "a": Integer(),
        "b": Float()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    rule = {
        "__rule__": Required("a", "b", "c"),
        "a": Integer(),
        "b": Float()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    rule = {
        "__rule__": Required("b"),
        "b": Float()
    }
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    

def test_Optional():
    config = {"a": None}
    rule = {"a": Optional(2).set_default_not_none()}
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    assert rule["a"].default == 2
    
    config = {"a": 2}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    