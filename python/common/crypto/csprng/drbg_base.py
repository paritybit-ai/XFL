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


import abc
from typing import Any, Generator, List, Union


class DRBGBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,
                 method: str,
                 entropy: Union[bytes, bytearray],
                 nonce: Union[bytes, bytearray] = b'',
                 additional_data: Union[bytes, bytearray] = b''):
        
        for i in [entropy, nonce, additional_data]:
            self._check_input_type(i)
        
        self.method = method
        self.entropy = entropy
        self.nonce = nonce
        self.additional_data = additional_data
        
    def _check_input_type(self, input: Any):
        if not isinstance(input, (bytes, bytearray)):
            raise TypeError(f"Expect type bytes or bytearray for DRBG input, got {type(input)}")
    
    @abc.abstractclassmethod
    def reseed(self, entropy: Union[bytes, bytearray], additional_data: Union[bytes, bytearray]):
        pass
        
    @abc.abstractclassmethod
    def gen(self, num_bytes: Union[List[int], int], additional_data: Union[bytes, bytearray]) -> bytearray:
        """normal mode"""
        pass
    
    @abc.abstractclassmethod
    def generator(self, num_bytes: Union[List[int], int], additional_data: Union[bytes, bytearray]) -> Generator:
        """generator mode"""
        pass
