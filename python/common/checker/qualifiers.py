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


from .base import Base


class OfBase(Base):
    def __init__(self, *args, default=None):
        super().__init__()
        self.candidates = args
        self.default = None
        
    def set_default(self, value):
        '''设置default值'''
        self.default = value
        return self

       
class OneOf(OfBase):
    '''
    只选择一个
    '''
    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = self.candidates[0]
    
    def set_default_index(self, idx):
        '''设置default的index'''
        self.default = self.candidates[idx]
        return self
    
    # def in_(self, values: list):
    #     def f(x):
    #         return x in values
    #     self.add_rule(f, f"is in {values}")
    #     return self
        
     
class SomeOf(OfBase):
    '''
    不放回的选多个(>=1, <=候选数)
    '''
    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = [self.candidates[0]]
        
    def set_default_indices(self, *idx):
        '''设置default的index，可以有多个'''
        self.default = [self.candidates[i] for i in idx]
        return self
    
    # def in_(self, values: list):
    #     def f(x):
    #         for v in x:
    #             if v not in values:
    #                 return False
    #         return True
    #     self.add_rule(f, f"is in {values}")
    #     return self
        
        
# repeatable, many              
class RepeatableSomeOf(OfBase):
    '''
    有放回的选多个
    '''
    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = [self.candidates[0]]
        
    def set_default_indices(self, *idx):
        '''设置default的index，可以有多个'''
        self.default = [self.candidates[i] for i in idx]
        return self
    
    # def in_(self, values: list):
    #     def f(x):
    #         for v in x:
    #             if v not in values:
    #                 return False
    #         return True
    #     self.add_rule(f, f"is in {values}")
    #     return self
        

class Required(OfBase):
    '''
    必须都存在
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.default = args
            
            
class Optional(OfBase):
    '''
    在dict的__rule__中表示该key可不存在，在list中表示list可为空，其他地方表示该值可为None。
    注：Optional只能接受一个参数。
    '''
    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        
    def set_default_not_none(self):
        '''设置default为非None值'''
        self.default = self.candidates[0]
        return self

        
# class Any(OfBase):
#     '''
#     任意值
#     '''
#     def __init__(self, default=None):
#         super().__init__(default=default)
        
#     def __eq__(self, __o: object) -> bool:
#         return True
    
#     def __hash__(self) -> int:
#         return int(''.join(map(lambda x: '%.3d' % ord(x), self.__name__() + "1234567890")))

