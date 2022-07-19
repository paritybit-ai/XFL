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


import random

import numpy as np
import pytest

from common.crypto.paillier.context import PaillierContext
from common.crypto.paillier.paillier import Paillier

data = [
    (True, None, True, -1),
    (True, 7, True, -1),
    (True, 7, False, 1),
    (False, None, True, -1),
    (False, 7, True, -1),
    (False, 7, False, 1)
]

data2 = [
    (True,),
    (False,)
]


class TestPaillier():
    def setup_class(self):
        self.context = PaillierContext.generate(2048)
        self.epsilon = 1e-5
    
    def teardown_class(self):
        pass
    
    def test_context(self):
        a = self.context.serialize(save_private_key=False)
        b = PaillierContext.deserialize_from(a)
        with pytest.raises(AssertionError):
            assert self.context == b

        a = self.context.serialize(save_private_key=True)
        b = PaillierContext.deserialize_from(a)
        assert self.context == b
    
    @pytest.mark.parametrize("djn_on, precision, is_batch, num_cores", data)
    def test_unary(self, djn_on, precision, is_batch,  num_cores):
        s = (50,)
        if is_batch:
            p1 = np.random.random(s).astype(np.float32) * 100 - 50
        else:
            p1 = random.random() * 100 - 50
        
        # encrypt
        c1 = Paillier.encrypt(self.context, p1, precision=precision, max_exponent=None, obfuscation=True, num_cores=num_cores)  
          
        # encrypt
        pub_context = self.context.to_public()
        c11 = Paillier.encrypt(pub_context, p1, precision=precision, max_exponent=None, obfuscation=True, num_cores=num_cores)
        
        # decrypt
        a = Paillier.decrypt(self.context, c1, num_cores=num_cores)
        assert (np.all(a - p1 < self.epsilon))
        
        b = Paillier.decrypt(self.context, c11, num_cores=num_cores)
        assert (np.all(b - p1 < self.epsilon))
        
        with pytest.raises(TypeError):
            Paillier.decrypt(pub_context, c1, num_cores=num_cores)
            
    @pytest.mark.parametrize("djn_on, precision, is_batch, num_cores", data)
    def test_binary(self, djn_on, precision, is_batch,  num_cores):
        s = (50,)
        if is_batch:
            p1 = np.random.random(s).astype(np.float32) * 100 - 50
            p2 = np.random.random(s).astype(np.float32) * 100 - 20
        else:
            p1 = random.random() * 100 - 50
            p2 = random.random() * 100 - 20
            
        # encrypt
        c1 = Paillier.encrypt(self.context, p1, precision=precision, max_exponent=None, obfuscation=True, num_cores=num_cores)
        c2 = Paillier.encrypt(self.context, p2, precision=precision, max_exponent=None, obfuscation=True, num_cores=num_cores)
        
        # sum
        if is_batch:
            c3 = sum(c1)
            a = Paillier.decrypt(self.context, c3, num_cores=num_cores)
            assert (a - sum(p1) < self.epsilon)
        
        # add
        c3 = c1 + c2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert (np.all(p3 - (p1 + p2) < self.epsilon))
        
        # sub
        c3 = c1 - c2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert(np.all(p3 - (p1 - p2) < self.epsilon))
        
        c21 = Paillier.ciphertext_from(self.context.to_public(), Paillier.serialize(c2))
        c3 = c1 - c21
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert(np.all(p3 - (p1 - p2) < self.epsilon))
        
        # add scalar
        c3 = c1 + p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p1 + p2) < self.epsilon)
        
        c3 = p2 + c1
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p1 + p2) < self.epsilon)
        
        # sub scalar
        c3 = c1 - p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p1 - p2) < self.epsilon)
        
        c3 = p2 - c1
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p2 - p1) < self.epsilon)
        
        c3 = c1 - p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p1 - p2) < self.epsilon)
        
        c3 = p2 - c1
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - (p2 - p1) < self.epsilon)
        
        # multiply
        c3 = c1 * p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 * p2 < self.epsilon)
        
        c3 = p2 * c1
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 * p2 < self.epsilon)
        
        c3 = c1 * p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 * p2 < self.epsilon)
        
        c3 = p2 * c1
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 * p2 < self.epsilon)
        
        # divide
        c3 = c1 / p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 / p2 < self.epsilon)
        
        c3 = c1 / p2
        p3 = Paillier.decrypt(self.context, c3, num_cores=num_cores)
        assert np.all(p3 - p1 / p2 < self.epsilon)
        
    @pytest.mark.parametrize("p1_is_batch", data2)
    def test_ciphertext(self, p1_is_batch):
        s = (3, 40)
        context = Paillier.context(2048)
        if p1_is_batch:
            p1 = np.random.random(s).astype(np.float32) * 100 - 50
        else:
            p1 = random.random() * 100 - 50
            
        context2 = Paillier.context_from(context.to_public().serialize())
            
        c1 = Paillier.encrypt(context2, p1, precision=None, max_exponent=None, obfuscation=True)
        s1 = Paillier.serialize(c1)
        
        c2 = Paillier.ciphertext_from(context, s1)
        p2 = Paillier.decrypt(context, c2)
        assert np.all(p1 - p2 < 1e-10)
    
    
# def test_paillier():
#     djn_on = True
#     num_cores = 1
#     precision = None
#     cal_paillier(djn_on, precision, True, True, 1e-10, num_cores)
#     # cal_paillier(djn_on, precision, True, False, 1e-10, num_cores)
#     # cal_paillier(djn_on, precision, False, True, 1e-10, num_cores)
#     cal_paillier(djn_on, precision, False, False, 1e-10, num_cores)
    

# def test_ciphertext(p1_is_batch=True):
#     print("------------ciphertext------------")
#     s = (3, 40)
#     context = Paillier.context(2048)
#     if p1_is_batch:
#         p1 = np.random.random(s).astype(np.float32) * 100 - 50
#     else:
#         # p1 = random.random() * 100 - 50
#         p1 = np.random.random((1,)).astype(np.float32) * 100 - 50
        
#     c1 = Paillier.encrypt(context, p1, precision=None, max_exponent=None, obfuscation=True)
#     start = time.time()
#     s1 = Paillier.serialize(c1)
#     print('ciphertext serialize:', time.time() - start)
    
#     start = time.time()
#     c2 = Paillier.ciphertext_from(context, s1)
#     print('ciphertext deserialize:', time.time() - start)
#     p2 = Paillier.decrypt(context, c2)
#     assert np.all(p1 - p2 < 1e-10)
    
    
# if __name__ == "__main__":
#     test_context()
#     test_paillier()
#     test_ciphertext(True)
