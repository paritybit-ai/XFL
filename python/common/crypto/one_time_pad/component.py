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


import pickle
from copy import deepcopy
from typing import Union

import numpy as np


class OneTimePadContext(object):
    def __init__(self,
                 modulus_exp: int = 64,
                 data_type: str = "torch.Tensor"):
        """Context includes modulus, plaintext data type, exponent for encoding and etc..

        Args:
            modulus_exp (int, optional): exponent(base 2) of modulus. Ciphertext will consists of integers
                module 2^modulus_exp. Defaults to 64.
            data_type (str, optional): plaintext type, supports "torch.Tensor" and "numpy.ndarray",
                or "torch" and "numpy" for short. Defaults to "torch.Tensor".

        Raises:
            ValueError: if modulus_exp not in [64, 128]
            ValueError: if data_type not in ["numpy.ndarray", "numpy", "torch.Tensor", "torch"]
        """
        if modulus_exp not in [64, 128]:
            raise ValueError(f"Supported modulus_exps are 64 and 128, got {modulus_exp}.")
        
        if data_type not in ["numpy.ndarray", "numpy", "torch.Tensor", "torch"]:
            raise ValueError(f"Supported data_types are 'numpy.ndarray', 'numpy', 'torch.Tensor', 'torch', got {data_type}.")
        
        if modulus_exp == 128:
            self.__exp = 64
        else:
            self.__exp = 32
            
        self.__modulus_exp = modulus_exp
        self.__scalar = 1 << self.__exp
        self.__modulus = 1 << modulus_exp
        
        if "numpy" in data_type:
            self.__data_type = np.ndarray
        elif "torch" in data_type:
            import torch
            self.__data_type = torch.Tensor
                    
        self.__security_strength = modulus_exp
        self.__encode_method = "fixed-point arithmetic"
            
    @property
    def exp(self):
        return self.__exp
    
    @property
    def modulus_exp(self):
        return self.__modulus_exp
    
    @property
    def scalar(self):
        return self.__scalar
    
    @property
    def modulus(self):
        return self.__modulus
    
    @property
    def data_type(self):
        return self.__data_type
    
    @property
    def security_strength(self):
        return self.__security_strength
    
    @property
    def encode_method(self):
        return self.__encode_method
    
    def __str__(self) -> str:
        out = "scalar: 1 << {}, modulus: 1 << {}, data_type: {}, security_strength: {}, encode_method: {}"
        out = out.format(self.exp, self.modulus_exp, self.data_type, self.security_strength, self.encode_method)
        return out
    
    def __eq__(self, other: object) -> bool:
        return self.__modulus_exp == other.modulus_exp
    
    @staticmethod
    def serialize(data) -> bytes:
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data) -> any:
        return pickle.loads(data)
    

class OneTimeKey(object):
    def __init__(self, key: Union[list[np.ndarray], np.ndarray], modulus_exp: int = 64):
        dtype = np.uint64 if modulus_exp == 64 else object
        modulus = (1 << modulus_exp)
        if isinstance(key, list):
            self.value = [np.array(np.mod(v, modulus)).astype(dtype) for v in key]
        else:
            self.value = np.array(np.mod(key, modulus)).astype(dtype)
            
    def __len__(self):
        return len(self.value)
    
    
class OneTimePadCiphertext(object):
    def __init__(self,
                 data: Union[list, np.ndarray, bytes],
                 context_: OneTimePadContext):
        """[summary]

        Args:
            data (Union[list, np.ndarray, bytes]): list or np.ndarray consists of integers, or picked object of them.
            context_ (OneTimePadContext): see OneTimePadContext.

        Raises:
            TypeError: if the type of data is not bytes, list or np.ndarray.
            TypeError: if the type of context_ is not OneTimePadContext or bytes.
        """
        if isinstance(context_, OneTimePadContext):
            self.__context = context_
        elif isinstance(context_, bytes):  
            self.__context = pickle.loads(context_)
        else:
            raise TypeError(f"Got context type {type(context_)}, supported types are 'OneTimePadContext', 'bytes'")
        
        dtype = np.uint64 if self.__context.modulus_exp == 64 else object
        
        if isinstance(data, bytes):
            self.__data = np.array(OneTimePadContext.deserialize(data), dtype=dtype)
        elif isinstance(data, list):
            self.__data = np.array(data, dtype=dtype)
        elif isinstance(data, (np.ndarray, np.float64, np.uint64)):  # , float, int)):
            self.__data = data.astype(dtype)
        else:
            raise TypeError(f"Got data type {type(data)}, supported types are 'list', 'np.ndarray', 'bytes'")

    def __str__(self):
        out = ', '.join([f"data: {self.__data}", "context: " + str(self.__context)])
        return out
        
    def __add__(self, other: object):
        if self.__context != other.__context:
            raise ValueError(f"Adding ciphertext with different context, {self.__context} vs {other.__context}")
        
        if self.__context.modulus_exp == 64:
            out = self.__data + other.__data
        else:
            out = np.array(np.mod(self.__data + other.__data, self.__context.modulus), dtype=object)
        out = OneTimePadCiphertext(out, self.__context)
        return out
            
    def __sub__(self, other: object):
        if self.__context != other.__context:
            raise ValueError(f"Subtracting ciphertext with different context, {self.__context} vs {other.__context}")
        
        if self.__context.modulus_exp == 64:
            out = self.__data - other.__data
        else:
            out = np.array(np.mod(self.__data - other.__data, self.__context.modulus), dtype=object)
        out = OneTimePadCiphertext(out, self.__context)
        return out
    
    @property
    def data(self):
        return self.__data
    
    @property
    def context_(self):
        return self.__context
    
    def serialize(self) -> bytes:
        """Pickle __data for transmission
        """
        return OneTimePadContext.serialize(self.__data)
        
    def decode(self):
        """Decode to plaintext when all the keys in the ciphertext are cancelled
        """
        if self.__data.shape == ():
            zero_shape = True
            data = np.array([self.__data], dtype=object)
        else:
            zero_shape = False
            data = self.__data.astype(object)
        
        idx = np.where(data > self.__context.modulus // 2)
        out = deepcopy(data)
        
        if len(idx[0]) != 0:
            out[idx] -= self.__context.modulus

        out /= self.__context.scalar
        
        if self.__context.data_type == np.ndarray:
            out = out.astype(np.float32)
        else:
            import torch
            out = torch.from_numpy(out.astype(np.float32))
        
        if zero_shape:
            return out[0]
        else:
            return out
