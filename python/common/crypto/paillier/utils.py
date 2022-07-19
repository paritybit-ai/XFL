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


import multiprocessing
import secrets
from typing import Optional

import gmpy2

MPZ = type(gmpy2.mpz())


def get_core_num(expected_core_num):
    max_cores = multiprocessing.cpu_count()
    if expected_core_num == -1:
        num_cores = max_cores
    else:
        num_cores = min(max(1, expected_core_num), max_cores)
    return num_cores


def mul(a, b):
    return gmpy2.mul(a, b)


def crt(mp, mq, p, q, q_inverse, n):
    """The Chinese Remainder Theorem, return the solution modulo n=pq.
    """
    u = gmpy2.mul(mp-mq, q_inverse) % p
    x = (mq + gmpy2.mul(u, q)) % n
    return int(x)


# def fmod(a, b):
#     return gmpy2.f_mod(a, b)


def mulmod(a, b, c):
    """
    return int: (a * b) % c
    """
    return gmpy2.mul(a, b) % c


def powmod(a: int, b: int, c: int) -> int:
    """
    return int: (a ** b) % c
    """
    if a == 1:
        return 1

    if max(a, b, c) < (1 << 64):
        return pow(a, b, c)
    else:
        return gmpy2.powmod(a, b, c)


def invert(a, b):
    """return int: x, where a * x == 1 mod b
    """
    x = gmpy2.invert(a, b)
    if x == 0:
        raise ZeroDivisionError('invert(a, b) no inverse exists')
    return x


def getprimeover(n, seed: Optional[int] = None):
    """return a random n-bit prime number #, p = 3 mod 4
    """
    if seed is not None:
        r = gmpy2.mpz(secrets.SystemRandom().getrandbits(n))
    else:
        r = gmpy2.mpz(secrets.SystemRandom(seed).getrandbits(n))
    r = gmpy2.bit_set(r, n - 1)
    
    out = gmpy2.next_prime(r)
    return out


def isqrt(n):
    """return the integer square root of N """
    return gmpy2.isqrt(n)


