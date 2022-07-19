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


import hashlib
import hmac
import warnings
from typing import Generator, List, Union

from .drbg_base import DRBGBase


class HMAC_DRBG(DRBGBase):
    """ 
    Implement HMAC_DRBG algorithm, defined by NIST SP 800-90A[1], Section 10.1.2.
    """
    '''NIST SP 800-90A Table 2'''
    __max_number_of_bytes_per_request = 1 << 16
    __maximum_number_of_requests_between_reseeds = 1 << 48
    # __max_bitlength = 1 << 35  # entropy, nonce, additonal_data should not exceed __max_length. Ignored.
        
    def __init__(self, method: str, entropy: Union[bytes, bytearray], nonce: Union[bytes, bytearray] = b'', additional_data: Union[bytes, bytearray] = b''):
        """
        Args:
            method (str): hash function, supported methods are SHA1, SHA224, SHA256, SHA284, SHA512
            entropy (Union[bytes, bytearray]): see [1].
            nonce (Union[bytes, bytearray], optional): see [1]. Defaults to b''.
            additional_data (Union[bytes, bytearray], optional): personalization string, see [1]. Defaults to b''.
        """
        super().__init__(method, entropy, nonce=nonce, additional_data=additional_data)
        
        method = method.lower()
        if method in ['sha1', 'sha224', 'sha256', 'sha384', 'sha512']:
            self.__hashfunc = getattr(hashlib, method)
        elif method == 'sm3':
            raise ValueError(f"Hash method {method} not supported in HMAC_DRBG, supported methods are SHA1, SHA224, SHA256, SHA284, SHA512")
        else:
            raise ValueError(f"Hash method {method} not supported in HMAC_DRBG, supported methods are SHA1, SHA224, SHA256, SHA284, SHA512")
        
        self.outlen_byte = self.__hashfunc().digest_size
        self.outlen_bit = 8 * self.outlen_byte
        """NIST SP 800-57 PART1 Table 3"""
        self.security_strength = self.outlen_bit // 2
        
        if len(entropy) < self.security_strength // 8:
            raise ValueError(f"Entropy's length is too short, should greater than {self.security_strength // 8} bytes.")
        
        self.__K = b'\x00' * self.outlen_byte
        self.__V = b'\x01' * self.outlen_byte
        
        self.__update(entropy + nonce + additional_data)
        
        self.__reseed_counter = 0
        self.acquire_reseed = False
        
        self.__buffer = bytearray()
        
    def __hmac(self, data: Union[bytes, bytearray] = b'') -> bytes:
        return hmac.new(self.__K, self.__V + data, self.__hashfunc).digest()
        
    def __update(self, additional_data: Union[bytes, bytearray] = b''):
        self.__K = self.__hmac(b'\x00' + additional_data)
        self.__V = self.__hmac()
        
        if len(additional_data) != 0:
            self.__K = self.__hmac(b'\x01' + additional_data)
            self.__V = self.__hmac()
            
    def __str__(self) -> str:
        s = "HMAC_DRBG: hash function {}, security strength {}"
        return s.format(self.method, self.security_strength)
    
    def reseed(self, entropy: Union[bytes, bytearray], additional_data: Union[bytes, bytearray] = b''):
        self.__update(entropy + additional_data)
        self.__reseed_counter = 0
        self.acquire_reseed = False
        
    def __gen(self, num_byte: int, additional_data: bytes = b''):
        """num_byte should <= __max_number_of_bytes_per_request
        """
        buffer = bytearray()
        
        if len(additional_data) != 0:
            self.__update(additional_data)
        
        while len(buffer) < num_byte:
            self.__V = self.__hmac()
            buffer += self.__V
            
        self.__update(additional_data)
        self.__buffer += buffer
    
    def generator(self, num_bytes: Union[List[int], int], additional_data: bytes = b'') -> Generator:
        """generator version, need to use next(...) to get result, or use a 'for' loop"""
        """
        NOTE:  The returned bytes of 
               calling generator several times and 
               calling generator one time by packing number of bytes to a list 
               are mainly DIFFERENT.
               
               for example,
               the bytes generate by generator([n1, n2, ...]) and {generator(n1), generator(n2), ...} are different.
               
               The returned bytes of
               calling generator one time with a list of number of bytes and
               calling generator one time with the sum of the same list
               will be the SAME.
               
               for example,
               the bytes generate by generator([n1, n2, ...]) and generator(sum([n1, n2, ...])) are the same.
        """
        self.__buffer.clear()
        
        if isinstance(num_bytes, int):
            num_bytes = [num_bytes]
            
        total_bytes = sum(num_bytes)
        quotient, remainder = divmod(total_bytes, self.__max_number_of_bytes_per_request)
        
        index = 0
        n_bytes = [self.__max_number_of_bytes_per_request] * quotient + [remainder] * (remainder > 0)
        
        for n in n_bytes:
            self.__gen(n, additional_data)
            self.__reseed_counter += 1
            
            while index < len(num_bytes) and len(self.__buffer) >= num_bytes[index]:
                out = self.__buffer[:num_bytes[index]]
                del self.__buffer[:num_bytes[index]]
                index += 1
                if self.__reseed_counter >= self.__maximum_number_of_requests_between_reseeds:
                    # almost never happens
                    self.acquire_reseed = True
                    warnings.warn("Max number of requests reached, HMAC_DRBG needs reseeding.")
                yield out
        
        self.__buffer.clear()
        
    def gen(self, num_bytes: Union[List[int], int], additional_data: bytes = b'') -> bytearray:
        """normal version, return result immediately"""
        self.__buffer.clear()
        
        is_integer = False
        if isinstance(num_bytes, int):
            is_integer = True
            num_bytes = [num_bytes]
            
        total_bytes = sum(num_bytes)
        quotient, remainder = divmod(total_bytes, self.__max_number_of_bytes_per_request)
        
        index = 0
        out = []
        n_bytes = [self.__max_number_of_bytes_per_request] * quotient + [remainder] * (remainder > 0)
        
        for n in n_bytes:
            self.__gen(n, additional_data)
            self.__reseed_counter += 1
            
            while index < len(num_bytes) and len(self.__buffer) >= num_bytes[index]:
                out.append(self.__buffer[:num_bytes[index]])
                del self.__buffer[:num_bytes[index]]
                index += 1
                if self.__reseed_counter >= self.__maximum_number_of_requests_between_reseeds:
                    # almost never happens
                    self.acquire_reseed = True
                    warnings.warn("Max number of requests reached, HMAC_DRBG needs reseeding.")
        
        self.__buffer.clear()
        if is_integer:
            out = out[0]
        return out
