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


from secrets import SystemRandom
from typing import List

from gmpy2 import powmod

from common.communication.gRPC.python.channel import DualChannel
from .contants import primes_hex


class DiffieHellman(object):
    """
    Implement Diffie-Hellman key exchange protocol.
    The security parameters are referenced from RFC 7919, , which are used in TLS 1.3.
    Shortest exponents are referenced from appendix A of RFC 7919.
    """
    supported_prime_bitlength = [2048, 3072, 4096, 6144, 8192]
    # symmetric_equivalent_strength
    supported_security_strength = [103, 125, 150, 175, 192]
    supported_shortest_exponents = [225, 275, 325, 375, 400]
    g = 2
    primes = [int(p.replace(' ', ''), 16) for p in primes_hex]

    def __init__(self,
                 fed_ids: List[str],
                 key_bitlength: int = 3072,
                 optimized: bool = True,
                 channel_name: str = "diffie_hellman"):
        
        try:
            self.index = self.supported_prime_bitlength.index(key_bitlength)
        except ValueError:
            message = "Input key_bitlength {} not supported! Need to be one of the {}"
            raise ValueError(message.format(key_bitlength, self.supported_prime_bitlength))
        
        self.chan = DualChannel(name=channel_name, ids=fed_ids)
        self.p = self.primes[self.index]
        self.key_bitlength = key_bitlength
        self.security_strength = self.supported_security_strength[self.index]
        self.shorest_exponent = self.supported_shortest_exponents[self.index]
        self.optimized = optimized

        self.lower_bound = 1 << (self.shorest_exponent - 1)
        
        if optimized:
            self.upper_bound = 1 << self.shorest_exponent
        else:
            self.upper_bound = self.p - 2

        self.rand_num_generator = SystemRandom()
    
    def __str__(self) -> str:
        s = "Diffie-Hellman key exchange: remote_id={}, key_bitlength={}," \
            "security_level={}, optimized={}"
        return s.format(self.chan.remote_id, self.key_bitlength,
                        self.security_strength, self.optimized)

    def exchange(self, out_bytes: bool = True):
        a = self.rand_num_generator.randint(self.lower_bound, self.upper_bound)
        g_power_a = powmod(self.g, a, self.p)
        index, g_power_b = self.chan.swap([self.index, g_power_a])
        if index != self.index:
            message = "Input key_bitlength are not the same! {} for local, {} from remote."
            raise ValueError(message.format(self.supported_prime_bitlength[self.index],
                                            self.supported_prime_bitlength[index]))
        secret_number = int(powmod(g_power_b, a, self.p))
        
        if out_bytes:
            secret_number = secret_number.to_bytes((secret_number.bit_length() + 7) // 8, 'big')

        return secret_number
