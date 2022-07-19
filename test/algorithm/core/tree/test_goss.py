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

from algorithm.core.tree.goss import Goss

normal_data = [
    (0.5, 0.4),
    (0.3, 0.7),
    (0, 0.5),
    (0.5, 0)
]

abnormal_data = [
    (0.5, 0.6),
    (1.2, 0.5),
    (0.5, 1.2),
    (-1, 0.5),
    (0.5, -1)
]

abnormal_data2 = [
    (0.0001, 0.0002)
]


class TestGoss():
    def setup_class(self):
        self.size = 2000
        self.g = np.random.random((self.size,))
        self.h = np.random.random((self.size,))     
    
    def teardown_class(self):
        pass
    
    @pytest.mark.parametrize("alpha, beta", normal_data)
    def test_normal(self, alpha, beta):
        goss = Goss(alpha, beta)
        selected_idx = goss.sampling(self.g)
        goss.update_gradients(self.g, self.h)
        assert len(selected_idx) == int(self.size * alpha) + int(self.size *beta)
        assert len(goss.rand_set_idx) == int(self.size * beta)
        assert len(np.unique(selected_idx)) == len(selected_idx)
        assert len(np.unique(goss.rand_set_idx)) == len(goss.rand_set_idx)
        
    @pytest.mark.parametrize("alpha, beta", abnormal_data)
    def test_abnormal_1(self, alpha, beta):
        with pytest.raises(ValueError):
            Goss(alpha, beta)
    
    @pytest.mark.parametrize("alpha, beta", abnormal_data2)
    def test_abnormal_2(self, alpha, beta):
        goss = Goss(alpha, beta)
        with pytest.raises(ValueError):
            goss.sampling(self.g)
    