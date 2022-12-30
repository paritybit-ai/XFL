import pytest

from common.checker.checker import check, cal_num_valid
from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


type_pairs = [
    (
        {
            "a": 1,
            "b": 'b',
            "c": 3.5,
            "d": True,
            "e": 43432.34124,
            "f": [343, "1232", {}]
            },
        {
            "a": Integer(4),
            "b": String("abc"),
            "c": Float(1.0),
            "d": Bool(False),
            "e": Any(),
            "f": All()
        })
]


@pytest.mark.parametrize("config, rule", type_pairs)
def test_types(config, rule):
    assert rule["a"].default == 4
    assert rule["b"].default == "abc"
    assert rule["c"].default == 1.0
    assert rule["d"].default is False
    assert rule["e"].default is None
    assert rule["f"].default is None
    
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": "a"}
    rules = [
        {"a": Integer()},
        {"a": Float()},
        {"a": Bool()}
    ]
    
    for rule in rules:
        r = check(config, rule)
        num_valid, num_total = cal_num_valid(r)
        assert num_valid < num_total
    
    config = {"a": 1}
    rule = {"a": String()}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    config = {"a": [{}]}
    rule = {"a": [Any()]}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    
def test_types_method():
    config = {"a": 3}
    rules = [
        {"a": Integer().ge(3).le(5)},
        {"a": Integer().gt(1).lt(5)},
        {"a": Integer().ge(4)},
        {"a": Integer().gt(4).lt(2)}
    ]
    
    for i, rule in enumerate(rules):
        r = check(config, rule)
        num_valid, num_total = cal_num_valid(r)
        if i in [0, 1]:
            assert num_valid == num_total
        elif i in [3, 4]:
            assert num_valid < num_total
            
    config = {"a": 3}
    rules = [
        {"a": Float().ge(3).le(5)},
        {"a": Float().gt(1).lt(5)},
        {"a": Float().ge(4)},
        {"a": Float().gt(4).lt(2)}
    ]
    
    for i, rule in enumerate(rules):
        r = check(config, rule)
        num_valid, num_total = cal_num_valid(r)
        if i in [0, 1]:
            assert num_valid == num_total
        elif i in [3, 4]:
            assert num_valid < num_total
            
    config = {"a": "324342"}
    rule = {"a": String().add_rule(lambda x: x[0] == 'a')}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
            
    config = {"a": {}}
    rule = {String(): {}}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": True}
    m = False
    rule = {"a": Bool().add_rule(lambda x: x == m)}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
            
    config = {"a": {}}
    rule = {Any(): {}}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    rule = {All(): {}}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    config = {"a": {}}
    rule = {"a": Any()}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
    
    rule = {"a": All()}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid == num_total
    
    rule = {"a": All().add_rule(lambda x: x)}
    r = check(config, rule)
    num_valid, num_total = cal_num_valid(r)
    assert num_valid < num_total
        



