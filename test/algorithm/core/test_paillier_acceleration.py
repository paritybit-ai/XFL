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
import pytest

from algorithm.core.paillier_acceleration import embed, umbed
from common.crypto.paillier.paillier import Paillier


def almost_equal(a, b, epsilon=1e-5):
    return np.all(np.abs(a - b) < epsilon)


p0 = np.random.random((1000,)) - 0.5
p1 = (np.random.random((1000,)) - 0.2) * 100


@pytest.mark.parametrize("p0, p1", [(p0, p1)])
def test_paillier_acceleration(p0, p1):
    context = Paillier.context(2048, True)
    embeded_p = embed([p0, p1])
    c = Paillier.encrypt(context, embeded_p)
    sum_c = np.sum(c)
    pa = Paillier.decrypt(context, sum_c, out_origin=True)
    p2, p3 = umbed([pa], 2)
    
    assert almost_equal(np.sum(p0), p2)
    assert almost_equal(np.sum(p1), p3)
