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
from functools import reduce
from random import randint
from secrets import token_hex

import numpy as np
import pytest
import torch

from common.crypto.one_time_pad.one_time_add import (OneTimeAdd,
                                                     OneTimePadCiphertext,
                                                     OneTimePadContext)


def almost_equal(a, b):
    if isinstance(a, np.ndarray):
        return np.all(a - b < 1e-4)
    elif isinstance(a, torch.Tensor):
        return torch.all(a - b < 1e-4)
    else:
        return a - b < 1e-4
    
    
# def correctness_scalar(modulus_exp, data_type, num_keys):
#     # random keys
#     key1 = [np.array(int(token_hex(modulus_exp//8), 16))]
    
#     for i in range(num_keys - 1):
#         key = int(token_hex(modulus_exp//8), 16)
#         key = np.array(key)
#         key1.append(key)
#     is_addition = randint(0, 1)
    
#     # random input
#     if "numpy" in data_type:
#         data = np.random.random(())
#     elif "torch" in data_type:
#         data = torch.rand(())
    
#     # context
#     context_ = OneTimePadContext(modulus_exp, data_type)
#     # encrypt
#     ciphertext = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
#     ciphertext2 = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=True)
#     assert pickle.dumps(ciphertext.data) == ciphertext2
    
#     # decrypt
#     plaintext = OneTimeAdd.decrypt(context_, ciphertext, key1, is_addition)
    
#     assert almost_equal(data, plaintext)
    
    
    
def correctness(data_shape, modulus_exp, data_type, num_keys):
    if data_shape == ():
        flatten_shape = 0
    else:
        flatten_shape = reduce(lambda x, y: x*y, data_shape)
    
    # random keys
    if flatten_shape == 0:
        key1 = [np.array(int(token_hex(modulus_exp//8), 16))]
    
        for i in range(num_keys - 1):
            key = int(token_hex(modulus_exp//8), 16)
            key = np.array(key)
            key1.append(key)
    else:
        key1 = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
        key1 = [np.array(key1).reshape(*data_shape)]
        
        for i in range(num_keys - 1):
            key = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
            key = np.array(key).reshape(*data_shape)
            key1.append(key)
        
    is_addition = [randint(0, 1) for i in range(len(key1))]
        
    # random input
    if "numpy" in data_type:
        data = np.random.random(data_shape)
    elif "torch" in data_type:
        data = torch.rand(data_shape)
        
    # context
    context_ = OneTimePadContext(modulus_exp, data_type)
    # encrypt
    ciphertext = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
    ciphertext2 = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=True)
    assert pickle.dumps(ciphertext.data) == ciphertext2
    
    # decrypt
    plaintext = OneTimeAdd.decrypt(context_, ciphertext, key1, is_addition)
    assert almost_equal(data, plaintext)
    
    # addition and subtraction
    # random input
    if "numpy" in data_type:
        data3 = np.random.random(data_shape)
    elif "torch" in data_type:
        data3 = torch.rand(data_shape)

    key3 = list(map(lambda x: np.array(-x), key1))
    ciphertext3 = OneTimeAdd.encrypt(context_, data3, key3, is_addition, serialized=False)
    ciphertext4 = OneTimeAdd.encrypt(context_, data3, key1, is_addition, serialized=False)
    
    c = ciphertext + ciphertext3
    plaintext = c.decode()
    assert almost_equal(data + data3, plaintext)
    
    c = ciphertext - ciphertext4
    plaintext = c.decode()
    assert almost_equal(data - data3, plaintext)
    
    if flatten_shape == 0:
        key2 = [np.array(int(token_hex(modulus_exp//8), 16))]
        
        for i in range(num_keys - 1):
            key = int(token_hex(modulus_exp//8), 16)
            key = np.array(key)
            key2.append(key)
    else:
        key2 = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
        key2 = [np.array(key2).reshape(*data_shape)]
    
        for i in range(num_keys - 1):
            key = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
            key = np.array(key).reshape(*data_shape)
            key2.append(key)
    
    ciphertext1 = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
    ciphertext2 = OneTimeAdd.encrypt(context_, data3, key2, is_addition, serialized=False)
    
    c = ciphertext1 + ciphertext2
    
    key3 = [key1[i] + key2[i] for i in range(len(key1))]

    p = OneTimeAdd.decrypt(context_, c, key3, is_addition)
    assert almost_equal(data + data3, p)
    
    ciphertext1 = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
    ciphertext2 = OneTimeAdd.encrypt(context_, data3, [-i for i in key2], is_addition, serialized=False)
    
    c = ciphertext1 + ciphertext2
    key3 = [key1[i] - key2[i] for i in range(len(key1))]

    p = OneTimeAdd.decrypt(context_, c, key3, is_addition)
    assert almost_equal(data + data3, p)
    
    
def test_correctness():
    data_shape_list = [(), (11,), (3, 5), (7, 10, 24)]
    modulus_exp_list = [64, 128]
    data_type_list = ["numpy.ndarray", "numpy", "torch.Tensor", "torch"]
    numpy_keys_list = [1, 3, 5]
    
    # for modulus_exp in modulus_exp_list:
    #     for data_type in data_type_list:
    #         for numpy_keys in numpy_keys_list:
    #             correctness_scalar(modulus_exp, data_type, numpy_keys)
    
    for data_shape in data_shape_list:
        for modulus_exp in modulus_exp_list:
            for data_type in data_type_list:
                for numpy_keys in numpy_keys_list:
                    print(data_shape, modulus_exp, data_type, numpy_keys)
                    correctness(data_shape, modulus_exp, data_type, numpy_keys)
                    
                    
def test_exception():
    modulus_exp = 128,
    data_type = "pandas"
    
    with pytest.raises(ValueError):
        OneTimePadContext(modulus_exp, data_type)
        
    with pytest.raises(ValueError):
        OneTimePadContext(modulus_exp, data_type)
    
    # ------------------------------------------------------------------------------
    
    modulus_exp = 128
    data_type = "numpy.ndarray"
    
    context_ = OneTimePadContext(modulus_exp, data_type)
    data = 'fdfdsfd'
    
    with pytest.raises(TypeError):
        OneTimePadCiphertext(data, context_)
        
    context_ = 54645654634
    data = np.array([2, 4])
    with pytest.raises(TypeError):
        OneTimePadCiphertext(data, context_)
        
    # ------------------------------------------------------------------------------
    
    key_shape = (3, 4)
    flatten_shape = 12
    key = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
    key = np.array(key).reshape(*key_shape)
    
    is_addition = [randint(0, 1) for i in range(len(key))]
    
    data_type = "torch.Tensor"
    context_ = OneTimePadContext(modulus_exp, data_type)
    
    data_shape = (4, 5)
    data = torch.rand(data_shape)
    with pytest.raises(ValueError):
        OneTimeAdd.encrypt(context_, data, key, is_addition, serialized=True)
        
    key_shape = (4, 5)
    flatten_shape = 20
    key = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
    key = np.array(key).reshape(*key_shape)
        
    # ------------------------------------------------------------------------------
    
    modulus_exp = 128
    data_type = 'numpy'
    key_shape = (3, 4)
    flatten_shape = 12
    key = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
    key = np.array(key).reshape(*key_shape)
    
    key_shape = (4, 5)
    flatten_shape = 20
    key1 = [int(token_hex(modulus_exp//8), 16) for i in range(flatten_shape)]
    key1 = np.array(key1).reshape(*key_shape)
    
    is_addition = randint(0, 1)
    
    data_shape = (4, 5)
    data = np.random.random(data_shape)

    context_ = OneTimePadContext(modulus_exp, data_type)
    ciphertext = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
    
    with pytest.raises(ValueError):
        OneTimeAdd.decrypt(context_, ciphertext, key, is_addition)
        
    key = [key, key]
    ciphertext = OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
    
    with pytest.raises(ValueError):
        OneTimeAdd.decrypt(context_, ciphertext, key, is_addition)
        
    # ------------------------------------------------------------------------------
        
    modulus_exp = 64
    context2 = OneTimePadContext(modulus_exp, data_type)
    ciphertext2 = OneTimeAdd.encrypt(context2, data, key1, is_addition, serialized=False)
    with pytest.raises(ValueError):
        ciphertext + ciphertext2
        
    # ------------------------------------------------------------------------------
        
    is_addition = [randint(0, 1) for i in range(len(key) + 1)]
    with pytest.raises(ValueError):
        OneTimeAdd.encrypt(context_, data, key1, is_addition, serialized=False)
        
