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


class Goss(object):
    """Gradient based one side sampling(GOSS) in LightGBM, obeyed to LightGBM original paper.
       Link: https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
    """
    def __init__(self, alpha: float, beta: float):
        """

        Args:
            alpha (float): sampling ratio of large gradient data.
            beta (float): sampling ratio of small gradient data.
        """
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError(f"alpha {alpha} and beta {beta} should >= 0 and <= 1.")

        if not 0 < alpha + beta <= 1:
            raise ValueError(f"alpha {alpha} + beta {beta} should between > 0 and <= 1")
        
        self.alpha = alpha
        self.beta = beta

    def sampling(self, g: np.ndarray) -> np.ndarray:
        """ Generate sample index

        Args:
            g (np.ndarray): gradients list, 1-d array.

        Returns:
            np.ndarray: selected sample index
        """
        # topN = a * len(i), randN = b * len(i)
        top_n, rand_n = int(self.alpha * len(g)), int(self.beta * len(g))
        
        # sorted = GetSortedIndices(abs(g))
        sorter = np.argsort(abs(g))[::-1]
        
        # topSet = sorted[1:topN]
        top_set_idx = sorter[:top_n]
        
        # randSet = RandomPick(sorted[topN:len(I)], randN)
        rand_set_idx = sorter[top_n:]
        rand_set_idx = np.random.choice(sorter[top_n:], rand_n, replace=False)
        self.rand_set_idx = rand_set_idx
        
        selected_idx = np.sort(np.concatenate([top_set_idx, rand_set_idx]))
        
        if len(selected_idx) == 0:
            raise ValueError("Length of selected sample is 0.")
        return selected_idx

    def update_gradients(self, g: np.ndarray, h: np.ndarray) -> None:
        if len(self.rand_set_idx) == 0:
            return
        g[self.rand_set_idx] *= (1 - self.alpha) / self.beta
        h[self.rand_set_idx] *= (1 - self.alpha) / self.beta
