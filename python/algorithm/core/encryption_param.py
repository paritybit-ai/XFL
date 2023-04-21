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


from typing import List, Optional, Union, Dict, Any

from common.utils.constants import CKKS, PAILLIER, PLAIN, OTP


# used by xgboost
class EncryptionParam(object):
    pass
        
        
class PaillierParam(object):
    def __init__(self, 
                 key_bit_size: int = 2048,
                 precision: Optional[int] = 7,
                 djn_on: bool = True,
                 parallelize_on: bool = False):
        self.method = PAILLIER
        self.key_bit_size = key_bit_size
        self.precision = precision
        self.djn_on = djn_on
        self.parallelize_on = parallelize_on
        

class CKKSParam(object):
    def __init__(self,
                 poly_modulus_degree: int = 8192,
                 coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
                 global_scale_bit_size: int = 40):
        self.method = CKKS
        self.poly_modulus_degress = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale_bit_size = global_scale_bit_size


class OTPParam(object):
    def __init__(self,
                 key_bitlength: int = 64,
                 data_type: str = "torch.Tensor",
                 key_exchange: Dict[str, Any] = None,
                 csprng: Dict[str, Any] = None):
        self.method = OTP
        self.key_bitlength = key_bitlength
        self.data_tyep = data_type
        self.key_exchange = key_exchange
        self.csprng = csprng


class PlainParam(object):
    def __init__(self):
        self.method = PLAIN
        

def get_encryption_param(method: str, params: Optional[dict] = None) -> Union[PlainParam, PAILLIER, CKKS]:
    if method == PLAIN:
        return PlainParam()
    elif method == PAILLIER:
        return PaillierParam(**params)
    elif method == CKKS:
        return CKKSParam(**params)
    elif method == OTP:
        return OTPParam(**params)
    else:
        raise ValueError(f"Encryption method {method} not supported.")
        
                 