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


import numpy as np

from common.crypto.csprng.drbg import get_drbg_inst


NAME = 'hmac_drbg'
METHOD = 'sha512'


def hex2bytes(a):
    return bytes.fromhex(a.replace(' ', ''))


def test_hmac_drbg_cross_validation():
    entropy = '000102 03040506'\
              '0708090A 0B0C0D0E 0F101112 13141516 1718191A 1B1C1D1E'\
              '1F202122 23242526 2728292A 2B2C2D2E 2F303132 33343536'
               
    nonce = '20212223 24252627'
    
    additional_data = '12345678'
    
    # ------------------------------------------------------------------------------

    drbg_g1 = get_drbg_inst(name=NAME, 
                            entropy=hex2bytes(entropy),
                            method=METHOD, 
                            nonce=hex2bytes(nonce),
                            additional_data=hex2bytes(additional_data))
    
    drbg_g2 = get_drbg_inst(name=NAME, 
                            entropy=hex2bytes(entropy),
                            method=METHOD, 
                            nonce=hex2bytes(nonce),
                            additional_data=hex2bytes(additional_data))
    
    drbg1 = get_drbg_inst(name=NAME, 
                          entropy=hex2bytes(entropy),
                          method=METHOD, 
                          nonce=hex2bytes(nonce),
                          additional_data=hex2bytes(additional_data))
    
    drbg2 = get_drbg_inst(name=NAME, 
                          entropy=hex2bytes(entropy),
                          method=METHOD, 
                          nonce=hex2bytes(nonce),
                          additional_data=hex2bytes(additional_data))
    
    num_bytes = [100] * 100 + [1000] * 1000
    
    # start = time.time()
    out1 = drbg1.gen(num_bytes)
    # print(time.time() - start)
    
    # start = time.time()
    out2 = drbg2.gen(sum(num_bytes))
    # print(time.time() - start)
    
    # start = time.time()
    g1 = drbg_g1.generator(num_bytes)
    out_g1 = []
    for o in g1:
        out_g1.append(o)
    # print(time.time() - start)
    
    num_bytes1 = sum(num_bytes)
    # start = time.time()
    g2 = drbg_g2.generator(num_bytes1)
    out_g2 = next(g2)
    # print(time.time() - start)

    assert out1[0] == out_g1[0]
    assert out1[-1] == out_g1[-1]
    
    assert np.all([a == b for a, b in zip(out1, out_g1)])
    assert out2 == out_g2
    assert out2 == b''.join(out_g1)
    
    
if __name__ == "__main__":
    test_hmac_drbg_cross_validation()