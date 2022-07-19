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


# import time
# from typing import Tuple

# import numpy as np
# import random
# from gmpy2 import mpz

# from fed_api import get_drbg_inst
    
    
# def hex2bytes(a):
#     return bytes.fromhex(a.replace(' ', ''))


# def split_bytes(x: bytes, out_shape: Tuple[int]):
#     if len(out_shape) == 0:
#         # return mpz(int(x, 16))
#         return mpz(int.from_bytes(x, 'big'))
#     elif len(out_shape) == 1:
#         a = len(x) // out_shape[0]
#         # return [mpz(int(x[a*i: a*(i+1)], 16)) for i in range(out_shape[0])]
#         return [mpz(int.from_bytes(x[a*i: a*(i+1)], 'big')) for i in range(out_shape[0])]
#     else:
#         a = len(x) // out_shape[0]
#         return [split_bytes(x[a*i: a*(i+1)], out_shape[1:]) for i in range(out_shape[0])]


# def benchmark():
#     # entropy = '000102 03040506'\
#     #           '0708090A 0B0C0D0E 0F101112 13141516 1718191A 1B1C1D1E'\
#     #           '1F202122 23242526 2728292A 2B2C2D2E 2F303132 33343536'
               
#     # nonce = '20212223 24252627'
    
#     # additional_data = ''
    
#     # # ------------------------------------------------------------------------------

#     # drbg = get_drbg_inst(name='hmac_drbg', 
#     #                      entropy=hex2bytes(entropy),
#     #                      method='sha512', 
#     #                      nonce=hex2bytes(nonce),
#     #                      additional_data=hex2bytes(additional_data))
    
#     # # first call to generate
    
#     # start = time.time()
#     # for i in range(1):
#     #     out = drbg.gen(num_bytes=16653*256*128//8)
#     # end = time.time()
    
#     # print(end - start)
    
#     # start = time.time()
#     # out1 = split_bytes(out, [16653, 256])
#     # end = time.time()
#     # print(end - start)
    
#     # start = time.time()
#     # b = np.frombuffer(bytes(out), np.uint8).reshape(16653, 256, 128//8)
#     # end = time.time()
#     # print(end - start)
    
#     # start = time.time()
#     # b = np.frombuffer(bytes(out), np.int64).reshape(16653, 256*2)
#     # end = time.time()
#     # print(end - start)
    
#     # start = time.time()
#     # bytes(out)
#     # print(time.time() - start)
    
#     # start = time.time()
#     # np.array(out1)
#     # end = time.time()
#     # print(end - start)
    
#     # print((end - start) / 1)
    
#     # start = time.time()
#     # out = drbg.generator(num_bytes=[16653*256*128//8]*1)
#     # for i in range(1):
#     #     next(out)
#     # end = time.time()
    
#     # print((end - start) / 1)
    
#     print("#########")
#     a = [random.randint(0, 2**128 - 1) for i in range(16653*256)]
#     b = [mpz(i) for i in a]
    
#     start = time.time()
#     a = np.array(a)
#     print(time.time() - start)
    
#     start = time.time()
#     b = np.array(b)
#     print(time.time() - start)
    
#     start = time.time()
#     # (a + a)
    
#     np.mod(a+a, 2**128)
#     print(time.time() - start)
    
#     start = time.time()
#     # b + b
#     np.mod(b+b, 2**128)
#     print(time.time() - start)
    

# if __name__ == "__main__":
#     benchmark()