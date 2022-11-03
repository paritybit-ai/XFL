"""
OneOf, AtLeastOneOf, RepeatableAtLeastOneOf, Required, Optional, Any

String, Bool, Integer, Float
"""

from typing import Callable


class Base(object):
    def __init__(self):
        self.default = None
        self.rules = []
        self.checked = []
        
    # def set_default(self, *value):
    #     if len(value) == 1:
    #         self.default = value[0]
    #     else:
    #         self.default = value
    #     return self
    
    def add_rule(self, rule: Callable, desp: str = ''):
        '''
        rule 可以接受一个参数，也可以接受两个参数。第一个参数表示当前位置的值，第二参数表示要检查的config。
        '''
        self.rules.append((rule, desp))
        return self
        
    # def check(self):  # TODO: add traceback
    #     self.checked = []
    #     for rule, desp in self.rules:
    #         try:
    #             is_pass = rule(self.value)
    #         except Exception:
    #             is_pass = False
    #         self.checked.append((is_pass, desp))
    #     return self.checked
    
    @property
    def __name__(self):
        return self.__class__.__name__
    
    def __hash__(self) -> int:
        return int(''.join(map(lambda x: '%.3d' % ord(x), self.__name__ + "1234567890")))
    
