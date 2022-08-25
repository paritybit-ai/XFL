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


def parse_category_param(df: pd.DataFrame,
                         col_index: str = "",
                         col_names: list[str] = [],
                         max_num_value: int = 0,
                         col_index_type: str = 'inclusive',
                         col_names_type: str = 'inclusive',
                         max_num_value_type: str = 'union') -> list[int]:
    """ Calculate column indexes and column names of category. The formulation is:

        features that column indexes are in col_index if col_index_type is 'inclusive' or not in col_index if col_index_type is 'exclusive'.

        union

        features that column names are in col_names if col_names_type is 'inclusive' or not in col_names if col_names_type is 'exclusive'.

        union if max_num_value_type is 'union' or intersect if max_num_value_type is 'intersection'

        features that number of different values is less equal than max_num_value.

    Args:
        df (pd.DataFrame): input dataframe.
        col_index (str): column index of features which are supposed to be (or not to be) a categorial feature. Defaults to "".
        col_names (list[str]): column names of features which are supposed to be (or not to be) a categorical feature. Defaults to [].
        max_num_value (int): if n <= max_num_value where n is the number of different values in a feature column, then the feature is supposed to be a category feature. Defalts to 0.
        col_index_type (str, optional): support 'inclusive' and 'exclusive'. Defaults to 'inclusive'.
        col_names_type (str, optional): support 'inclusive' and 'exclusive'. Defaults to 'inclusive'.
        max_num_value_type (str, optional): support 'intersection' and 'union'. Defaults to 'union'.

    Returns:
        list[int]: list of categorial feature column indexes.

    Note:
        col_index is count from the first column of features, not the input table.
        col_index support single value and slice. For example, a vaild form of col_index is "2, 4:8, -7, -10:-7", where "4:8" means "4,5,6,7",
        vaild form of col_names is like ["wage", "age"].
    """
    res = []

    if col_index != "":
        index1 = _parse_index(col_index, len(df.columns))

        if col_index_type == 'inclusive':
            res += index1
        elif col_index_type == 'exclusive':
            res += list(set(range(len(df.columns))) - set(index1))
        else:
            raise ValueError(
                f"col_index_type {col_index_type} not valid, need to be one of the 'inclusive' and 'exclusive'.")

    if col_names != []:
        index2 = _parse_names(col_names, df.columns.to_list())

        if col_names_type == 'inclusive':
            res += index2
        elif col_names_type == 'exclusive':
            res += list(set(range(len(df.columns))) - set(index2))
        else:
            raise ValueError(
                f"col_names_type {col_names_type} not valid, need to be one of the 'inclusive' and 'exclusive'.")

    res = list(set(res))

    if max_num_value > 0:
        if max_num_value_type == "union":
            num_unique = df.nunique().to_numpy()
            index3 = list(np.where(num_unique <= max_num_value)[0])
            res += index3
        elif max_num_value_type == "intersection":
            col_selection = [False for i in range(len(df.columns))]
            for i in res:
                col_selection[i] = True
            df_category = df.iloc[:, col_selection]
            num_unique = df_category.nunique().to_numpy()
            index3 = list(np.where(num_unique <= max_num_value)[0])
            res = list(map(lambda x: res[x], index3))
        else:
            raise ValueError(
                f"max_num_value_type {max_num_value_type} not valid, need to be one of the 'union' and 'intersect'.")
            
    res = list(set(res))
    return res


def _parse_index(index: str, num_cols: int) -> list[int]:
    ''' index form is "1, 3:5, 4, 8:11'''
    res = []
    index_list = index.replace(' ', '').split(',')
    
    for value in index_list:
        if ':' in value:
            left, right = value.split(':')
            if left == "":
                left = 0
            if int(left) < 0:
                left = min(max(0, num_cols + int(left)), num_cols)
            
            if right == "":
                right = num_cols
            if int(right) < 0:
                right = min(max(0, num_cols + int(right)), num_cols)
                
            res += [i for i in range(int(left), int(right))]
        else:
            value = int(value)
            if abs(value) >= num_cols:
                raise ValueError(f"Column index {value} is greater equal than the column size {num_cols}")
            
            if value < 0:
                value += num_cols
            res.append(value)
            
    res = list(set(res))
    return res


def _parse_names(names: list[str], valid_names: list[str]) -> list[int]:
    res = []
    name_list = [item.strip() for item in names]

    for name in name_list:
        try:
            i = valid_names.index(name)
            res.append(i)
        except ValueError as e:
            raise ValueError(f"Column name {name} not found: {e}")

    res = list(set(res))
    return res

