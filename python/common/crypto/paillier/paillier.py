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
import secrets
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import zstd
from pathos.pools import ProcessPool

from .context import PaillierContext
from .encoder import PaillierEncoder
from .utils import MPZ, crt, get_core_num, invert, mul, mulmod, powmod

"""
We suggest the plaintext to be a numpy.ndarray of dtype np.float32 to get more accurate result.
key_length ~ symmetric_equivalent_security_strength
1024 ~ 80
2048 ~ 112
3072 ~ 128
"""


class RawCiphertext:
    def __init__(self, value, exp):
        self.value = value
        self.exp = exp


class PaillierCiphertext(object):
    def __init__(self,
                 context: PaillierContext,
                 raw_ciphertext: MPZ,
                 exponent: int) -> None:
        self.__context = context
        self.__c = raw_ciphertext
        self.__exp = exponent
        
    @property
    def context(self):
        return self.__context
    
    @property
    def raw_ciphertext(self):
        return self.__c
    
    @property
    def exponent(self):
        return self.__exp

    def serialize(self, compression: bool = True) -> bytes:
        out = pickle.dumps(RawCiphertext(self.__c, self.__exp))
        if compression:
            out = zstd.compress(out)
        return out
    
    @classmethod
    def deserialize_from(cls, context: PaillierContext, data: bytes, compression: bool = True):
        if compression:
            data = zstd.decompress(data)
        unpickled_data = pickle.loads(data)
        return PaillierCiphertext(context, unpickled_data.value, unpickled_data.exp)
    
    def _decrease_exponent_to(self, new_exponent: int):
        scalar = 1 << (self.__exp - new_exponent)
        raw_ciphertext = self._raw_mul(self.raw_ciphertext, 
                                       scalar,
                                       self.context.min_value_for_negative,
                                       self.context.n,
                                       self.context.n_square)
        return raw_ciphertext
    
    def __add__(self, other):
        """[]
        Attention!: No obfuscation is applyed when adding a scalar, one need to explicitly
        execute 'obfuscate' method if needed.
        """
        if isinstance(other, PaillierCiphertext):
            return self._add_encrypted(other)
        elif isinstance(other, (int, float)):
            ciphertext = Paillier.encrypt(self.context,
                                          other,
                                          precision=None,
                                          max_exponent=None,
                                          obfuscation=False)
            
            return self._add_encrypted(ciphertext)
        else:
            raise TypeError(f"Adding data of type {type(other)} not supported.")
        
    def _add_encrypted(self, other):
        if self.context.to_public() != other.context.to_public():
            raise ValueError("Adding two ciphertext with different keys.")
        
        if self.exponent > other.exponent:
            new_exponent = other.exponent
            raw_a = self._decrease_exponent_to(new_exponent)
            raw_b = other.raw_ciphertext
        else:
            new_exponent = self.exponent
            raw_a = self.raw_ciphertext
            if self.exponent < other.exponent:
                raw_b = other._decrease_exponent_to(new_exponent)
            else:
                raw_b = other.raw_ciphertext
            
        raw_sum = self._raw_add(raw_a, raw_b, self.context.n_square)
        return PaillierCiphertext(self.context, raw_sum, new_exponent)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(other * (-1))
    
    def __rsub__(self, other):
        return ((-1) * self).__add__(other)
    
    def __mul__(self, scalar: Union[int, float]):
        if isinstance(scalar, PaillierCiphertext):
            raise TypeError("Cannot multiply one ciphertext with another ciphertext, try multiply a scalar.")

        exponent = PaillierEncoder.cal_exponent(scalar, precision=None)
        encoded_scalar = PaillierEncoder.encode_single(self.context, scalar, exponent)
        raw_ciphertext = self._raw_mul(self.raw_ciphertext, 
                                       encoded_scalar,
                                       self.context.min_value_for_negative,
                                       self.context.n,
                                       self.context.n_square)
        return PaillierCiphertext(self.context, raw_ciphertext, exponent + self.exponent)
        
    def __rmul__(self, scalar: Union[int, float]):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]):
        return self.__mul__(1 / scalar)
    
    def _raw_add(self, c1: MPZ, c2: MPZ, n_square: MPZ) -> MPZ:
        return mulmod(c1, c2, n_square)
    
    def _raw_mul(self, c1: MPZ, p2: MPZ, min_value_for_negative: MPZ, n: MPZ, n_square: MPZ) -> MPZ:
        '''0 <= p2 < n, optimize for -1'''
        if self.__context.is_private():
            p_square = self.__context.p_square
            q_square = self.__context.q_square
            phi_p2 = self.__context.phi_p2
            phi_q2 = self.__context.phi_q2
            q2_inverse_p2 = self.__context.q2_inverse_p2
            
            def crt_(mp_square, mq_square):
                return crt(mp_square, mq_square, p_square, q_square, q2_inverse_p2, n_square)
            
            def crt_powmod(c, p):
                mp_square = powmod(c % p_square, p % phi_p2, p_square)
                mq_square = powmod(c % q_square, p % phi_q2, q_square)
                return crt_(mp_square, mq_square)
            
            if p2 >= min_value_for_negative:
                if p2 + 102400 > n:  # abs(p2) is small
                    out = powmod(c1, n - p2, n_square)
                    return invert(out, n_square)
                
                invert_c1 = invert(c1, n_square)
                return crt_powmod(invert_c1, n - p2)
            else:
                return crt_powmod(c1, p2)
        else:
            if p2 >= min_value_for_negative:
                invert_c1 = invert(c1, n_square)
                return powmod(invert_c1, n - p2, n_square)
            else:
                return powmod(c1, p2, n_square)
    
    def obfuscate(self):
        n = self.__context.n
        n_square = self.__context.n_square
        
        if self.__context.djn_on:
            if self.__context.is_private():
                a = secrets.SystemRandom().randrange(1, self.__context.djn_exp_bound)
                phi_p2 = self.__context.phi_p2
                phi_q2 = self.__context.phi_q2
                ep = self.__context.ep
                eq = self.__context.eq
                p_square = self.__context.p_square
                q_square = self.__context.q_square
                q2_inverse_p2 = self.__context.q2_inverse_p2
                
                def crt_(mp_square, mq_square):
                    return crt(mp_square, mq_square, p_square, q_square, q2_inverse_p2, n_square)
                
                mp_square = powmod(self.__context.h_pow_n_modp2, a % phi_p2, p_square)
                mq_square = powmod(self.__context.h_pow_n_modq2, a % phi_q2, q_square)
                obfuscator = crt_(mp_square, mq_square)
            else:
                a = secrets.SystemRandom().randrange(1, self.__context.djn_exp_bound)
                obfuscator = powmod(self.__context.h_pow_n, a, n_square)
        else:
            if self.__context.is_private():
                r = secrets.SystemRandom().randrange(1, n)
                ep = self.__context.ep
                eq = self.__context.eq
                p_square = self.__context.p_square
                q_square = self.__context.q_square
                q2_inverse_p2 = self.__context.q2_inverse_p2
                
                def crt_(mp_square, mq_square):
                    return crt(mp_square, mq_square, p_square, q_square, q2_inverse_p2, n_square)
                
                mp_square = powmod(r % p_square, ep, p_square)
                mq_square = powmod(r % q_square, eq, q_square)
                obfuscator = crt_(mp_square, mq_square)
            else:
                r = secrets.SystemRandom().randrange(1, n)
                obfuscator = powmod(r, n, n_square)
        self.__c = mulmod(self.__c, obfuscator, n_square)
        return self


class Paillier(object):
    @staticmethod
    def context(key_bit_size: int = 2048, djn_on: bool = False):
        return PaillierContext.generate(key_bit_size, djn_on)
    
    @staticmethod
    def context_from(data: bytes):
        return PaillierContext.deserialize_from(data)
    
    @staticmethod
    def serialize(data: Union[np.ndarray, PaillierCiphertext], compression: bool = True) -> bytes:
        """data is PaillierCiphertext or a numpy.ndarray consists of PaillierCiphertext"""
        if isinstance(data, PaillierCiphertext):
            return data.serialize(compression)
        
        def f(x):
            return RawCiphertext(x.raw_ciphertext, x.exponent)
        
        f1 = np.vectorize(f)
        out = f1(data)
        out = pickle.dumps(out)
        if compression:
            out = zstd.compress(out)
        return out
    
    @staticmethod
    def ciphertext_from(context: PaillierContext, data: bytes, compression: bool = True):
        if compression:
            data = zstd.decompress(data)
        unpickled_data = pickle.loads(data)
            
        def f(x):
            return PaillierCiphertext(context, x.value, x.exp)
        
        f1 = np.vectorize(f, otypes=[PaillierCiphertext])
        out = f1(unpickled_data)
        return out
    
    @staticmethod
    def _encrypt_single(data: Union[int, float],
                        context: PaillierContext,
                        precision: Optional[int] = None,
                        max_exponent: Optional[int] = None,
                        obfuscation: bool = True) -> MPZ:
        exponent = PaillierEncoder.cal_exponent(data, precision)
        if max_exponent is not None:
            exponent = min(exponent, max_exponent)
        encoded_data = PaillierEncoder.encode_single(context, data, exponent)
        raw_ciphertext = (mul(context.n, encoded_data) + 1) % context.n_square
        out = PaillierCiphertext(context, raw_ciphertext, exponent)
        if obfuscation:
            return out.obfuscate()
        return out
        
    @classmethod
    def encrypt(cls,
                context: PaillierContext,
                data: Union[int, float, np.ndarray],
                precision: Optional[int] = None,
                max_exponent: Optional[int] = None,
                obfuscation: bool = True,
                num_cores: int = -1) -> Union[PaillierCiphertext, np.ndarray]:
        """[summary]

        Args:
            context (PaillierContext): [description]
            data (Union[int, float, np.ndarray]): [description]
            precision (Optional[int], optional): [description]. Defaults to None.
            max_exponent (Optional[int], optional): [description]. Defaults to None.
            obfuscation (bool, optional): [description]. Defaults to True.
            num_cores (int, optional): how many cores are used for encryption. \
                Defaults to 1. Set to inf to use all the cores.

        Raises:
            TypeError: [description]

        Returns:
            Union[PaillierCiphertext, np.ndarray]: [description]
        """
        if isinstance(data, np.ndarray):
            if num_cores == 1:
                def f1(x):
                    c = cls._encrypt_single(x, context, precision, max_exponent, obfuscation)
                    return c
                f2 = np.vectorize(f1, otypes=[PaillierCiphertext])
                ciphertext = f2(data)
            else:
                num_cores = get_core_num(num_cores)
                partial_encrypt = partial(cls._encrypt_single,
                                          context=context, 
                                          precision=precision,
                                          max_exponent=max_exponent,
                                          obfuscation=obfuscation)
            
                s = data.shape
                data_flatten = data.reshape(-1)
                with ProcessPool(num_cores) as pool:
                    ciphertext = np.array(pool.map(partial_encrypt, data_flatten)).reshape(s)
                # pool.terminate()        
        elif isinstance(data, (int, float)):
            # ciphertext = cls._encrypt_single(context, data, precision, max_exponent, obfuscation)
            ciphertext = cls._encrypt_single(data, context, precision, max_exponent, obfuscation)
        else:
            raise TypeError(f"Unsupported data type {type(data)}, accepted types are 'np.ndarray', 'int', 'float'.")
        return ciphertext
    
    @staticmethod
    def _decrypt_single(data: PaillierCiphertext,
                        context: PaillierContext) -> float:
        if not isinstance(data, PaillierCiphertext):
            return data
        
        p, q = context.p, context.q
        q_inverse_p = context.q_inverse_p
        p_square = context.p_square
        q_square = context.q_square
        hp = context.hp
        hq = context.hq
        n = context.n
        
        def l_func(x, m):
            return (x - 1) // m

        def crt_(mp, mq):
            return crt(mp, mq, p, q, q_inverse_p, n)

        mp = l_func(powmod(data.raw_ciphertext, p - 1, p_square), p)
        mp = mulmod(mp, hp, p)
        mq = l_func(powmod(data.raw_ciphertext, q - 1, q_square), q)
        mq = mulmod(mq, hq, q)
        encoded_number = crt_(mp, mq)
        
        out = PaillierEncoder.decode_single(context, encoded_number, data.exponent)
        return out
    
    @classmethod
    def decrypt(cls,
                context: PaillierContext,
                data: Union[PaillierCiphertext, np.ndarray],
                dtype: str = 'float',
                num_cores: int = -1,
                out_origin: bool = False):
        if not context.is_private():
            raise TypeError("Try to decrypt a paillier ciphertext by a public key.")

        if isinstance(data, np.ndarray):
            if num_cores == 1:
                def f1(x):
                    p = cls._decrypt_single(x, context)
                    return p
                f2 = np.vectorize(f1)
                out = f2(data)
            else:
                num_cores = get_core_num(num_cores)
                partial_decrypt = partial(cls._decrypt_single, context=context)
                
                s = data.shape
                data_flatten = data.reshape(-1)
                with ProcessPool(num_cores) as pool:
                    out = np.array(pool.map(partial_decrypt, data_flatten)).reshape(s)       
            
            if not out_origin:
                if dtype == 'float':
                    out = out.astype(np.float32)
                elif dtype == 'int':
                    out = out.astype(np.int32)
                else:
                    warnings.warn(f"dtype {dtype} not supported.")
                    out = out.astype(np.float32)   
        elif isinstance(data, PaillierCiphertext):
            out = cls._decrypt_single(data, context)
            
            if not out_origin:
                if 'float' in dtype:
                    out = float(out)
                elif 'int' in dtype:
                    out = int(out)
                else:
                    warnings.warn(f"dtype {dtype} not supported.")
                    out = float(out)
        else:
            raise TypeError(f"Unsupported data type {type(data)}, accepted types are 'np.ndarray', 'int', 'float'.")
        return out
    
    @classmethod
    def obfuscate(cls, ciphertext: Union[PaillierCiphertext, np.ndarray]):
        if isinstance(ciphertext, np.ndarray):
            def f(c):
                r = c.obfuscate()
                return r
            f2 = np.vectorize(f, otypes=[PaillierCiphertext])
            ciphertext = f2(ciphertext)
        elif isinstance(ciphertext, PaillierCiphertext):
            ciphertext = ciphertext.obfuscate()
        else:
            raise TypeError(f"Unsupported raw ciphertext type {type(ciphertext)}")
        return ciphertext
