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


from common.crypto.csprng.drbg import get_drbg_inst


"""
HMAC_DRBG
Tese cases provide by:
https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/HMAC_DRBG.pdf

Test method=sha256
Ignore sha1, sha224, sha384, sha512
"""


def hex2bytes(a):
    return bytes.fromhex(a.replace(' ', ''))


def test_hmac_drbg_correctness():
    
    entropy = '000102 03040506'\
              '0708090A 0B0C0D0E 0F101112 13141516 1718191A 1B1C1D1E'\
              '1F202122 23242526 2728292A 2B2C2D2E 2F303132 33343536'
               
    nonce = '20212223 24252627'
    
    additional_data = ''
    
    # ------------------------------------------------------------------------------

    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(additional_data))
    
    # first call to generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = 'D67B8C17 34F46FA3 F763CF57 C6F9F4F2'\
                'DC1089BD 8BC1F6F0 23950BFC 56176352 08C85012 38AD7A44'\
                '00DEFEE4 6C640B61 AF77C2D1 A3BFAA90 EDE5D207 406E5403'
    
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second call to generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = '8FDAEC20 F8B42140 7059E358 8920DA7E'\
                'DA9DCE3C F8274DFA 1C59C108 C1D0AA9B 0FA38DA5 C792037C'\
                '4D33CD07 0CA7CD0C 5608DBA8 B8856546 39DE2187 B74CB263'
    
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    additional_data1 = '606162 63646566'\
                       '6768696A 6B6C6D6E 6F707172 73747576 7778797A 7B7C7D7E'\
                       '7F808182 83848586 8788898A 8B8C8D8E 8F909192 93949596'
                       
    additional_data2 = 'A0A1A2 A3A4A5A6'\
                       'A7A8A9AA ABACADAE AFB0B1B2 B3B4B5B6 B7B8B9BA BBBCBDBE'\
                       'BFC0C1C2 C3C4C5C6 C7C8C9CA CBCCCDCE CFD0D1D2 D3D4D5D6'
                       
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(additional_data))
    
    # first call to generate
    out = drbg.generator(num_bytes=512//8, additional_data=hex2bytes(additional_data1))
    
    rand_val1 = '41878735 8135419B 93813353 5306176A'\
                'FB251CDD 2BA37988 59B566A0 5CFB1D68 0EA92585 6D5B84D5'\
                '6ADAE870 45A6BA28 D2C908AB 75B7CC41 431FAC59 F38918A3'
    
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second call to generate
    out = drbg.generator(num_bytes=512//8, additional_data=hex2bytes(additional_data2))
    
    rand_val2 = '7C067BDD CA817248 23D64C69 829285BD'\
                'BFF53771 6102C188 2E202250 E0FA5EF3 A384CD34 A20FFD1F'\
                'BC91E0C5 32A8A421 BC4AFE3C D47F2232 3EB4BAE1 A0078981'
    
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    personalzation_str = '404142 43444546'\
                         '4748494A 4B4C4D4E 4F505152 53545556 5758595A 5B5C5D5E'\
                         '5F606162 63646566 6768696A 6B6C6D6E 6F707172 73747576'
                         
    # ------------------------------------------------------------------------------
                       
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(personalzation_str))
    
    # first call to generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = '0DD9C855 89F357C3 89D6AF8D E9D734A9'\
                '17C771EF 2D8816B9 82596ED1 2DB45D73 4A626808 35C02FDA'\
                '66B08E1A 369AE218 F26D5210 AD564248 872D7A28 784159C3'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second call to generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val2 = '46B4F475 6AE715E0 E51681AB 2932DE15'\
                '23BE5D13 BAF0F458 8B11FE37 2FDA37AB E3683173 41BC8BA9'\
                '1FC5D85B 7FB8CA8F BC309A75 8FD6FCA9 DF43C766 0B221322'
                
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(personalzation_str))
    
    # first call to generate
    out = drbg.generator(num_bytes=512//8, additional_data=hex2bytes(additional_data1))
    
    rand_val1 = '1478F29E 94B02CB4 0D3AAB86 245557CE'\
                '13A8CA2F DB657D98 EFC19234 6B9FAC33 EA58ADA2 CCA432CC'\
                'DEFBCDAA 8B82F553 EF966134 E2CD139F 15F01CAD 568565A8'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # first call to generate
    out = drbg.generator(num_bytes=512//8, additional_data=hex2bytes(additional_data2))
    
    rand_val2 = '497C7A16 E88A6411 F8FCE10E F56763C6'\
                '1025801D 8F51A743 52D682CC 23A0A8E6 73CAE032 28939064'\
                '7DC683B7 342885D6 B76AB1DA 696D3E97 E22DFFDD FFFD8DF0'
       
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(additional_data))
    
    entropy1 = '808182 83848586'\
               '8788898A 8B8C8D8E 8F909192 93949596 9798999A 9B9C9D9E'\
               '9FA0A1A2 A3A4A5A6 A7A8A9AA ABACADAE AFB0B1B2 B3B4B5B6'
    
    entropy2 = 'C0C1C2 C3C4C5C6'\
               'C7C8C9CA CBCCCDCE CFD0D1D2 D3D4D5D6 D7D8D9DA DBDCDDDE'\
               'DFE0E1E2 E3E4E5E6 E7E8E9EA EBECEDEE EFF0F1F2 F3F4F5F6'
               
    # first reseed
    drbg.reseed(entropy=hex2bytes(entropy1))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = 'FABD0AE2 5C69DC2E FDEFB7F2 0C5A31B5'\
                '7AC938AB 771AA19B F8F5F146 8F665C93 8C9A1A5D F0628A56'\
                '90F15A1A D8A613F3 1BBD65EE AD5457D5 D26947F2 9FE91AA7'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second reseed
    drbg.reseed(entropy=hex2bytes(entropy2))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val2 = '6BD925B0 E1C232EF D67CCD84 F722E927'\
                'ECB46AB2 B7400147 77AF14BA 0BBF53A4 5BDBB62B 3F7D0B9C'\
                '8EEAD057 C0EC754E F8B53E60 A1F434F0 5946A8B6 86AFBC7A'
                
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(additional_data))
    
    # first reseed
    drbg.reseed(entropy=hex2bytes(entropy1), additional_data=hex2bytes(additional_data1))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = '085D57AF 6BABCF2B 9AEEF387 D531650E'\
                '6A505C54 406AB37A 52899E0E CAB3632B 7A068A28 14C6DF6A'\
                'E532B658 D0D9741C 84775FEE 45B684CD BDC25FBC B4D8F310'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second reseed
    drbg.reseed(entropy=hex2bytes(entropy2), additional_data=hex2bytes(additional_data2))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val2 = '9B219FD9 0DE2A08E 493405CF 874417B5'\
                '826770F3 94481555 DC668ACD 96B9A3E5 6F9D2C32 5E26D47C'\
                '1DFCFC8F BF86126F 40A1E639 60F62749 342ECDB7 1B240DC6'
                
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(personalzation_str))
    
    # first reseed
    drbg.reseed(entropy=hex2bytes(entropy1))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = 'D8B67130 714194FF E5B2A35D BCD5E1A2'\
                '9942AD5C 68F3DEB9 4ADD9E9E BAD86067 EDF04915 FB40C391'\
                'EAE70C65 9EAAE7EF 11A3D46A 5B085EDD 90CC72CE A989210B'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second reseed
    drbg.reseed(entropy=hex2bytes(entropy2))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val2 = '8BBA71C2 583F2530 C259C907 84A59AC4'\
                '4D1C8056 917CCF38 8788102D 73824C6C 11D5D63B E1F01017'\
                'D884CD69 D9334B9E BC01E7BD 8FDF2A8E 52572293 DC21C0E1'
                
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    # ------------------------------------------------------------------------------
    
    drbg = get_drbg_inst(name='hmac_drbg', 
                         entropy=hex2bytes(entropy),
                         method='sha256', 
                         nonce=hex2bytes(nonce),
                         additional_data=hex2bytes(personalzation_str))
    
    # first reseed
    drbg.reseed(entropy=hex2bytes(entropy1), additional_data=hex2bytes(additional_data1))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val1 = '44D78BBC 3EB67C59 C22F6C31 003D212A'\
                '7837CCD8 4C438B55 150FD013 A8A78FE8 EDEA81C6 72E4B8DD'\
                'C8183886 E69C2E17 7DF574C1 F190DF27 1850F8CE 55EF20B8'
                
    assert next(out).hex() == rand_val1.replace(' ', '').lower()
    
    # second reseed
    drbg.reseed(entropy=hex2bytes(entropy2), additional_data=hex2bytes(additional_data2))
    
    # generate
    out = drbg.generator(num_bytes=512//8)
    
    rand_val2 = '917780DC 0CE9989F EE6C0806 D6DA123A'\
                '18252947 58D4E1B5 82687231 780A2A9C 33F1D156 CCAD3277'\
                '64B29A4C B2690177 AE96EF9E E92AD0C3 40BA0FD1 203C02C6'\
                
    assert next(out).hex() == rand_val2.replace(' ', '').lower()
    
    
if __name__ == "__main__":
    test_hmac_drbg_correctness()