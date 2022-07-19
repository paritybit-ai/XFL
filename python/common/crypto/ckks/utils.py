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


# important

supported_poly_modulus_degree = [1024, 2048, 4096, 8192, 16384, 32768]

max_coeff_modulus_bitlength = {
    1024: 27,
    2048: 54,
    4096: 109,
    8192: 218,
    16384: 438,
    32768: 881
}

security_table = [
    ['security_level', 'poly_moduls_degree', 'max_coeff_modulus_bitlength', 'suggested_coeff_mod_bit_sizes'],
    [128, 1024, 27, None],
    [128, 2048, 54, None],
    [128, 4096, 109, [40, 20, 40]],
    [128, 8192, 218, [60, 40, 40, 60]],
    [128, 16384, 438, None],
    [128, 32768, 881, None],
    [192, 1024, 19, None],
    [192, 2048, 37, None],
    [192, 4096, 75, None],
    [192, 8192, 152, None],
    [192, 16384, 305, None],
    [192, 32768, 611, None],
    [256, 1024, 14, None],
    [256, 2048, 29, None],
    [256, 4096, 58, None],
    [256, 8192, 118, None],
    [256, 16384, 237, None],
    [256, 32768, 476, None],
]