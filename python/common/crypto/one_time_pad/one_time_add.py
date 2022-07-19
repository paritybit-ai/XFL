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


import warnings
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch

from .component import OneTimePadCiphertext, OneTimePadContext


class OneTimeAdd(object):
    """Provide encrypt and decrypt method for one-time-add algorithm
    """
    @staticmethod
    def context(modulus_exp: int = 128,
                data_type: str = "torch.Tensor"):
        return OneTimePadContext(modulus_exp, data_type)
    
    @staticmethod
    def ciphertext(data: Union[list, np.ndarray, bytes],
                   context_: OneTimePadContext):
        return OneTimePadCiphertext(data, context_)
    
    @staticmethod
    def _xcrypt(context_: OneTimePadContext,
                data: Union[np.ndarray, torch.Tensor],
                one_time_key: List[np.ndarray],
                is_addition: Union[List[bool], bool] = True,
                is_decrypt: bool = False) -> Union[List[OneTimePadCiphertext], OneTimePadCiphertext]:
        """ Function for implementing encryption and decryption.
        is_addition: same length as one_time_key, means to add or to subtract the key. default to True;
        if is_decrypt is False, return a numpy array of integers;
        if is_decrypt is True, return a numpy array of float numbers.
        """
        if isinstance(is_addition, (bool, int)):
            is_addition = [is_addition] * len(one_time_key)
        elif len(is_addition) != len(one_time_key):
            raise ValueError(f"Length of is_additon ({len(is_addition)}) and one_time_key ({len(one_time_key)}) not match.")

        if data.shape == ():
            zero_shape = True
            data = np.array([data])
        else:
            zero_shape = False
        
        if not is_decrypt:
            out = np.mod(np.trunc(data * context_.scalar), context_.modulus).astype(object)
        else:
            out = deepcopy(data)

        for i in range(len(one_time_key)):
            if is_addition[i] - is_decrypt:
                out = np.mod(np.trunc(out + one_time_key[i]), context_.modulus).astype(object)
            else:
                out = np.mod(out - one_time_key[i], context_.modulus).astype(object)
                    
        if is_decrypt:
            idx = np.where(out > context_.modulus // 2)
            out[idx] -= context_.modulus
            out /= context_.scalar
            
        if zero_shape:
            out = np.array(out[0]).astype(object)
        return out
    
    @classmethod
    def encrypt(cls,
                context_: OneTimePadContext,
                data: Union[np.ndarray, torch.Tensor],
                one_time_key:  Union[List[np.ndarray], np.ndarray],
                is_addition: Union[List[bool], bool] = True,
                serialized: bool = False) -> Union[OneTimePadCiphertext, bytes]:
        """Encrypt the data to a ciphertext

        Args:
            context_ (OneTimePadContext): see OneTimePadContext.
            data (Union[np.ndarray, torch.Tensor]): plaintext to encrypt.
            one_time_key (Union[List[np.ndarray], np.ndarray]): a key for addition/subtraction, or a list of keys, 
                                the ciphertext is plaintext +/- key[0] +/- key[1] +/- key[2] +/- ...
            is_addition (Union[List[bool], bool], optional): same length as one_time_key, means to add or to subtract the key.
                                Defaults to True.
            serialized (bool, optional): it is convenient to set it to true if the ciphertext needs to
                                be sent by the network right after the encryption. Defaults to False.

        Raises:
            ValueError: if shape of data is different from shape of one_time_key or one_time_key[0].
            
        Warnings:
            if context_.data_type is different from the type of data, which means the type of plaintext
                after decryption will be different from the type of plaintext before encryption.

        Returns:
            Union[OneTimePadCiphertext, bytes]: if serialized is False, return OneTimePadCiphertext,
                else return pickled ciphertext(numpy.ndarray of integers).
        """
        if isinstance(one_time_key, np.ndarray):
            one_time_key = [one_time_key.astype(object)]
        else:
            one_time_key = [np.array(key).astype(object) for key in one_time_key]
            
        if data.shape != one_time_key[0].shape:
            raise ValueError(f"Input data's shape {data.shape} and one_time_key's shape {one_time_key[0].shape} not match.")
            
        if not isinstance(data, context_.data_type) and not isinstance(data, np.float64):
            warnings.warn(f"Input data type {type(data)} and context_.data_type {context_.data_type} are different.")

        if isinstance(data, torch.Tensor):
            data = data.numpy().astype(object)
            
        out = cls._xcrypt(context_, data, one_time_key, is_addition, False)
        
        if not serialized:
            out = OneTimePadCiphertext(out, context_)
        else:
            out = OneTimePadContext.serialize(out)
        return out
    
    @classmethod
    def decrypt(cls,
                context_: OneTimePadContext,
                ciphertext: Union[OneTimePadCiphertext, bytes],
                one_time_key: Union[List[np.ndarray], np.ndarray],
                is_addition: Union[List[bool], bool] = True) -> Union[np.ndarray, torch.Tensor]:
        """Decrypt the ciphertext to a plaintext

        Args:
            context_ (OneTimePadContext): see OneTimePadContext.
            ciphertext (Union[OneTimePadCiphertext, bytes]): result of cls.encrypt(...) method.
            one_time_key (Union[List[np.ndarray], np.ndarray]): the same as it is in cls.encrypt(...).
            is_addition (Union[List[bool], bool]): the same as it is in cls.encrypt(...).
            
        Raises:
            ValueError: if the shape of ciphertext.data is different from the shape of one_time_key.

        Returns:
            Union[np.ndarray, torch.Tensor]: numpy.ndarray or torch.Tensor of float32, depend on context_.data_type
        """
        if isinstance(one_time_key, np.ndarray):
            one_time_key = [one_time_key.astype(object)]
        elif isinstance(one_time_key, list):
            one_time_key = [np.array(key).astype(object) for key in one_time_key]
            
        if isinstance(ciphertext, bytes):
            ciphertext = OneTimePadContext.deserialize(ciphertext)
        
        if ciphertext.data.shape != one_time_key[0].shape:
            raise ValueError(f"Input ciphertext's shape {ciphertext.data.shape} and one_time_key's shape {one_time_key[0].shape} not match.")
        
        out = cls._xcrypt(context_, ciphertext.data, one_time_key, is_addition, True)
        
        if context_.data_type == np.ndarray:
            out = out.astype(np.float32)
        else:
            out = torch.from_numpy(out.astype(np.float32))
        return out
