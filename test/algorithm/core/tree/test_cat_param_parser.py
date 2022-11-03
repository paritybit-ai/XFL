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
import pandas as pd
import pytest

from algorithm.core.tree.cat_param_parser import parse_category_param


def test_parse_category_param():
    df = pd.DataFrame(np.arange(70).reshape(7, 10),
                      columns=list(map(chr, range(97, 107))))
    df.iloc[:, 9] = 0
    df.iloc[:, 6] = 1
    columns_name = df.columns.to_list()
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_index_type='inclusive')
    assert res == [0, 3, 4]
    
    res = parse_category_param(df,
                               col_index="-7, -5:-3",
                               col_index_type='inclusive')
    assert res == [3, 5, 6]
    
    with pytest.raises(ValueError):
        res = parse_category_param(df,
                                   col_index="-11, -5:-3",
                                   col_index_type='inclusive')
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_index_type='exclusive')
    assert res == list(set(range(10)) - set({0, 3, 4}))
    
    res = parse_category_param(df,
                               col_names=["g", "a"],
                               col_index_type='inclusive')
    res_name = [columns_name[i] for i in res]
    assert set(res_name) == set(["g", "a"])
    
    res = parse_category_param(df,
                               col_names=["g", "a"],
                               col_names_type='exclusive')
    res_name = [columns_name[i] for i in res]
    assert set(res_name) == (set(list(map(chr, range(97, 107)))) - set(["g", "a"]))
    
    res = parse_category_param(df,
                               max_num_value=6,
                               max_num_value_type="intersection")
    res_name = [columns_name[i] for i in res]
    assert set(res_name) == set()
    
    res = parse_category_param(df,
                               max_num_value=6,
                               max_num_value_type="union")
    assert set(res) == set({6, 9})

    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='inclusive',
                               col_names_type='inclusive',
                               max_num_value_type="union")
    assert set(res) == set({0, 3, 4, 6, 9})
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='inclusive',
                               col_names_type='inclusive',
                               max_num_value_type="intersection")
    assert set(res) == set({6})
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='exclusive',
                               col_names_type='inclusive',
                               max_num_value_type="union")
    assert set(res) == set({1, 2, 4, 5, 6, 7, 8, 9})
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='inclusive',
                               col_names_type='exclusive',
                               max_num_value_type="union")
    assert set(res) == set(range(10))
    
    res = parse_category_param(df,
                               col_index="0, 3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='inclusive',
                               col_names_type='exclusive',
                               max_num_value_type="intersection")
    assert set(res) == set({9})
    
    res = parse_category_param(df,
                               col_index="-4, -3:5",
                               col_names=["g", "e"],
                               max_num_value=6,
                               col_index_type='inclusive',
                               col_names_type='inclusive',
                               max_num_value_type="intersection")
    assert set(res) == set({6})
