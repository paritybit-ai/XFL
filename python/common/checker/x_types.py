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


class String(Base):
    def __init__(self, default: str = ""):
        super().__init__()
        self.default = default
        # self.filled = None

        
class Bool(Base):
    def __init__(self, default: bool = True):
        super().__init__()
        self.default = default
        
        
class Integer(Base):
    def __init__(self, default: int = 0):
        super().__init__()
        self.default = default
        
    def gt(self, value):
        self.add_rule(lambda x: x > value, f"greater than {value}")
        return self
        
    def ge(self, value):
        self.add_rule(lambda x: x >= value, f"greater equal than {value}")
        return self
        
    def lt(self, value):
        self.add_rule(lambda x: x < value, f"less than {value}")
        return self
        
    def le(self, value):
        self.add_rule(lambda x: x <= value, f"less equal than {value}")
        return self


class Float(Base):
    def __init__(self, default: float = 0):
        super().__init__()
        self.default = default
        
    def gt(self, value):
        self.add_rule(lambda x: x > value, f"greater than {value}")
        return self
        
    def ge(self, value):
        self.add_rule(lambda x: x >= value, f"greater equal than {value}")
        return self
        
    def lt(self, value):
        self.add_rule(lambda x: x < value, f"less than {value}")
        return self
        
    def le(self, value):
        self.add_rule(lambda x: x <= value, f"less equal than {value}")
        return self
        
        
class Any(Base):
    '''
    任意值
    '''
    def __init__(self, default=None):
        super().__init__()
        self.default = default
        
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, (list, dict, tuple)):
            return False
        else:
            return True
    
    def __hash__(self) -> int:
        return int(''.join(map(lambda x: '%.3d' % ord(x), self.__name__ + "1234567890")))
    

class All(Base):
    def __init__(self, default=None):
        super().__init__()
        self.default = default
        
    def __eq__(self, __o: object) -> bool:
        return True
    
    def __hash__(self) -> int:
        return int(''.join(map(lambda x: '%.3d' % ord(x), self.__name__ + "1234567890")))