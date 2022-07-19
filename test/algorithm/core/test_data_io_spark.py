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


# import pytest
# import numpy as np
# import os
# import pyspark.pandas as pd
# import shutil
# from algorithm.core.data_io_spark import ValidationNumpyDataset,CsvReader



# def prepare_data():
#     case_df = pd.DataFrame({
#         'x0': np.arange(10),
#         'x1': np.arange(10),
#         'x2': 2 * np.arange(10) - 10,
#     })
#     case_df['y'] = np.where(
#         case_df['x1'] + case_df['x2'] > 10, 1, 0)
#     case_df[['y', 'x0', 'x1', 'x2']].to_csv(
#         "/opt/dataset/unit_test/test_data_io.csv", index=True
#     )
    

# @pytest.fixture(scope="module", autouse=True)
# def env():
#     os.chdir("python")
#     if not os.path.exists("/opt/dataset/unit_test"):
#         os.makedirs("/opt/dataset/unit_test")
#     prepare_data()
#     yield
#     if os.path.exists("/opt/dataset/unit_test"):
#         shutil.rmtree("/opt/dataset/unit_test")

# @pytest.fixture()  
# def data():
#     yield CsvReader("/opt/dataset/unit_test/test_data_io.csv", has_id=True, has_label=True)

# class TestCsvReader():
    
#     def test_features(data):
#         assert data.features() == np.array([np.arange(10),np.arange(10),2 * np.arange(10) - 10])
#         assert data.features("dataframe") == pd.DataFrame({
#         'x0': np.arange(10),
#         'x1': np.arange(10),
#         'x2': 2 * np.arange(10) - 10,
#     })
    

#     def test_label(data):
#         assert data.label() == np.arange(10)
#         assert data.label("dataframe").to_numpy().astype(np.float32) == np.arange(10)


#     def test_col_names(data):
#         assert data.col_names == ['y', 'x0', 'x1', 'x2']
    

#     def test_feature_names():
#         assert data.feature_names  == ['x0', 'x1', 'x2']
    
#     def test_label_name():
#         assert data.label_name == 'y'



# class TestValidationNumpyDataset():
    
#     def test_builtins():
#         data =  np.arange(10)
#         label = np.array([1,1,1,1,1,0,0,0,0,0])
#         batch_size = 4
#         dataset = ValidationNumpyDataset(data, label, batch_size)
#         for i,(x,y) in enumerate(dataset):
#             if i == 0:
#                 assert (x,y) == (np.array([0,1,2,3]),np.array([1,1,1,1]))
#             elif i == 1:
#                 assert (x,y) == (np.array([4,5,6,7]),np.array([1,0,0,0]))
#             elif i == 2:
#                 assert (x,y) == (np.array([8,9]),np.array([0,0]))
