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


import math
import pickle
import secrets
import warnings
from typing import Optional, Union

import gmpy2

from .utils import MPZ, getprimeover, invert, mul, powmod


class PaillierContext(object):
    def init(self, 
             p: Optional[Union[int, MPZ]] = None, 
             q: Optional[Union[int, MPZ]] = None, 
             n: Optional[Union[int, MPZ]] = None,
             djn_h_pow_n: Optional[Union[int, MPZ]] = None):
        if n is None and (p is None or q is None):
            raise ValueError("Insufficient parameters.")
        
        if p is not None and q is not None:
            self.__p = p
            self.__q = q
            self.__n = mul(p, q)
            if n is not None and self.__n != n:
                warnings.warn(f"Input n {n} not equal to p * q {self.__n}, use p * q instead.")
                
            self.q_inverse_p = invert(q, p)
            self.p_square = mul(p, p)
            self.q_square = mul(q, q)
            self.q2_inverse_p2 = invert(self.q_square, self.p_square)
            self.hp = self._h_function(p, self.p_square)
            self.hq = self._h_function(q, self.q_square)
            self.phi_p2 = mul(p, p-1)
            self.phi_q2 = mul(q, q-1)
            self.ep = self.__n % self.phi_p2
            self.eq = self.__n % self.phi_q2
            self.__is_private = True
        else:
            self.__n = n
            self.__is_private = False
            
        if djn_h_pow_n:
            self.h_pow_n = djn_h_pow_n
            self.djn_exp_bound = pow(2, self.__n.bit_length() // 2)
            self.djn_on = True
            if self.__is_private:
                self.h_pow_n_modp2 = self.h_pow_n % self.p_square
                self.h_pow_n_modq2 = self.h_pow_n % self.q_square
        else:
            self.djn_on = False
            
        self.n_square = pow(self.__n, 2)
        self.max_value_for_positive = self.__n // 3
        self.min_value_for_negative = self.__n - self.max_value_for_positive
        return self
    
    @classmethod
    def generate(cls, key_bit_size: int = 2048, djn_on: bool = False):
        p, q = cls._generate_paillier_private_key(key_bit_size, djn_on)
        if djn_on:
            n = mul(p, q)
            x = secrets.SystemRandom().getrandbits(n.bit_length())
            x = gmpy2.bit_set(x, n.bit_length() - 1)
            h = -pow(x, 2)
            h_pow_n = powmod(h, n, pow(n, 2))
            return PaillierContext().init(p, q, djn_h_pow_n=h_pow_n)
        else:
            return PaillierContext().init(p, q)
        
    @property
    def p(self):
        if self.__is_private:
            return self.__p
        return None
    
    @property
    def q(self):
        if self.__is_private:
            return self.__q
        return None
    
    @property
    def n(self):
        return self.__n
    
    def is_private(self):
        return self.__is_private
    
    def _copy_public_from(self, other):
        self.__n = other.n
        self.__is_private = False
        self.n_square = other.n_square
        self.max_value_for_positive = other.max_value_for_positive
        self.min_value_for_negative = other.min_value_for_negative
        self.djn_on = other.djn_on
        if self.djn_on:
            self.h_pow_n = other.h_pow_n
            self.djn_exp_bound = other.djn_exp_bound
    
    def to_public(self):
        if not self.__is_private:
            return self
        pub_context = PaillierContext()
        pub_context._copy_public_from(self)
        return pub_context
        
    @staticmethod
    def _generate_paillier_private_key(n_length: int = 2048, djn_on: bool = False, seed: Optional[int] = None): # djn_on: bool = True, 
        """
        Paillier-DJN: 
        Damgård I, Jurik M, Nielsen J B. 
        A generalization of Paillier’s public-key system with applications to electronic voting[J]. 
        International Journal of Information Security, 2010, 9(6): 371-385.
        """
        p, q = None, None
        
        if djn_on:
            def f(x, y):
                return (x == y) or (math.gcd(p-1, q-1) != 2)
        else:
            def f(x, y):
                return x == y
        
        if seed is None:
            while f(p, q):
                p = getprimeover(n_length // 2)
                q = getprimeover(n_length // 2)
        else:
            i = 1
            while f(p, q):
                p = getprimeover(n_length // 2, seed + i * 3)
                q = getprimeover(n_length // 2, seed + i * 5)
                i += 1
        return p, q

    def serialize(self, save_private_key: bool = True):
        if save_private_key and self.__is_private:
            out = pickle.dumps((self.__p, self.__q))
        else:
            out = pickle.dumps((self.__n,))
        return out
    
    @classmethod
    def deserialize_from(cls, data: bytes):  
        unpickled_data = pickle.loads(data)
        
        if len(unpickled_data) == 1:
            return PaillierContext().init(n=unpickled_data[0])
        elif len(unpickled_data) == 2:
            return PaillierContext().init(p=unpickled_data[0], q=unpickled_data[1])
        else:
            return ValueError("The unpickled data should be a tuple contains 1 or 2 big integers.")
        
    def __eq__(self, other):
        if id(self) == id(other):
            return True
        
        if self.p != other.p or self.q != other.q or self.n != other.n:
            return False
        return True
    
    def __hash__(self):
        if self.__is_private:
            return hash((self.__p, self.__q))
        else:
            return hash(self.__n)
    
    def __str__(self):
        if self.__is_private:
            return f"PaillierContext: p = {int(self.__p)}, q = {int(self.__q)}, n = {int(self.__n)}"
        else:
            return f"PaillierContext: n = {int(self.__n)}"
        
    def _l_function(self, x, p):
        return (x - 1) // p
    
    def _h_function(self, x, xsquare):
        return invert(self._l_function(powmod(self.__n + 1, x - 1, xsquare), x), x)
