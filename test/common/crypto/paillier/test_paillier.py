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
import pickle
import pytest

from common.crypto.paillier.context import PaillierContext
from common.crypto.paillier.paillier import Paillier, PaillierCiphertext, RawCiphertext

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
        
        with pytest.raises(ValueError):
            PaillierContext().init()
        
        PaillierContext().init(3, 5, 10)
        
        p, q = PaillierContext._generate_paillier_private_key(2048)

    @pytest.mark.parametrize("djn_on, precision, is_batch, num_cores", data)
    def test_unary(self, djn_on, precision, is_batch,  num_cores):
        s = (50,)
        if is_batch:
            p1 = np.random.random(s).astype(np.float32) * 100 - 50
        else:
            p1 = random.random() * 100 - 50

        # encrypt
        c1 = Paillier.encrypt(self.context, p1, precision=precision,
                              max_exponent=None, obfuscation=True, num_cores=num_cores)

        # encrypt
        pub_context = self.context.to_public()
        c11 = Paillier.encrypt(pub_context, p1, precision=precision,
                               max_exponent=None, obfuscation=True, num_cores=num_cores)

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
        c1 = Paillier.encrypt(self.context, p1, precision=precision,
                              max_exponent=None, obfuscation=True, num_cores=num_cores)
        c2 = Paillier.encrypt(self.context, p2, precision=precision,
                              max_exponent=None, obfuscation=True, num_cores=num_cores)

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

        c21 = Paillier.ciphertext_from(
            self.context.to_public(), Paillier.serialize(c2))
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

        c1 = Paillier.encrypt(context2, p1, precision=None,
                              max_exponent=None, obfuscation=True)
        s1 = Paillier.serialize(c1)

        c2 = Paillier.ciphertext_from(context, s1)
        p2 = Paillier.decrypt(context, c2)
        assert np.all(p1 - p2 < 1e-10)

    @pytest.mark.parametrize("compression", [True, False])
    def test_rest(self, compression):
        context = Paillier.context(2048)
        p1 = random.random() * 100 - 50
        c1 = Paillier.encrypt(context, p1, precision=7,
                              max_exponent=None, obfuscation=True)
        serialized_c1 = c1.serialize(compression)
        c1_2 = PaillierCiphertext.deserialize_from(context, serialized_c1, compression)
        assert c1.raw_ciphertext == c1_2.raw_ciphertext
        assert c1.exponent == c1_2.exponent
        
        with pytest.raises(TypeError):
            c1 + "342"
        
        context2 = Paillier.context(2048)
        p2 = random.random() * 100 - 50
        c2 = Paillier.encrypt(context2, p2, precision=7,
                              max_exponent=None, obfuscation=True)
        with pytest.raises(ValueError):
            c1 + c2
        
        with pytest.raises(TypeError):
            c1 * c1
            
        pub_context = context.to_public()
        c1 = Paillier.encrypt(pub_context, p1, precision=7,
                              max_exponent=None, obfuscation=True)
        p1_1 = Paillier.decrypt(context, c1)
        assert abs(p1 - p1_1) < 1e-5
        c1 = Paillier.obfuscate(c1)
        p1_1 = Paillier.decrypt(context, c1)
        assert abs(p1 - p1_1) < 1e-5
        
        context = Paillier.context(2048, djn_on=True)
        c1 = Paillier.encrypt(context, p1, precision=7,
                              max_exponent=None, obfuscation=True)
        p1_1 = Paillier.decrypt(context, c1)
        assert abs(p1 - p1_1) < 1e-5
        
        pub_context = context.to_public()
        c1 = Paillier.encrypt(pub_context, p1, precision=7,
                              max_exponent=None, obfuscation=True)
        p1_1 = Paillier.decrypt(context, c1)
        assert abs(p1 - p1_1) < 1e-5
        
        c1 = Paillier.encrypt(pub_context, p1, precision=7,
                              max_exponent=20, obfuscation=False)
        p1_1 = Paillier.decrypt(context, c1, num_cores=1)
        assert abs(p1 - p1_1) < 1e-5
        
        c1 = Paillier.encrypt(pub_context, p1, precision=7,
                              max_exponent=20, obfuscation=True, num_cores=1)
        p1_1 = Paillier.decrypt(context, c1)
        assert abs(p1 - p1_1) < 1e-5
        
        with pytest.raises(TypeError):
            Paillier.encrypt(pub_context, "123", precision=7,
                             max_exponent=None, obfuscation=True)
        
        p3 = 3
        c3 = Paillier.encrypt(pub_context, p3, precision=7,
                              max_exponent=None, obfuscation=True)

        p3_1 = Paillier.decrypt(context, c3)
        assert p3_1 == p3
        c3 = Paillier.obfuscate(c3)
        p3_1 = Paillier.decrypt(context, c3, dtype='int')
        assert np.all(p3_1 == p3)
        p3_1 = Paillier.decrypt(context, c3, dtype='i64')
        assert np.all(p3_1 == p3)
        
        p4 = np.array([2, 3], dtype=np.int32)
        c4 = Paillier.encrypt(pub_context, p4, precision=7,
                              max_exponent=None, obfuscation=True)
        p4_1 = Paillier.decrypt(context, c4, dtype='int')
        assert np.all(p4_1 == p4)
        p4_1 = Paillier.decrypt(context, c4, dtype='i64')
        assert np.all(p4_1 == p4)
        
        c4 = Paillier.encrypt(pub_context, p4, precision=7,
                              max_exponent=None, obfuscation=True, num_cores=1)
        p4_1 = Paillier.decrypt(context, c4)
        assert np.all(p4_1 == p4)
        p4_1 = Paillier.decrypt(context, c4, num_cores=1)
        assert np.all(p4_1 == p4)
        c4 = Paillier.obfuscate(c4)
        p4_1 = Paillier.decrypt(context, c4)
        assert np.all(p4_1 == p4)
        
        assert 123 == Paillier._decrypt_single(123, context)
        
        with pytest.raises(TypeError):
            Paillier.decrypt(context, 123)
            
        with pytest.raises(TypeError):
            Paillier.obfuscate(123)
        

