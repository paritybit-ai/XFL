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

import pytest
import numpy as np
import pandas as pd
import os 
import pathlib
import gzip
import zipfile
import tarfile
import shutil
import urllib.request
from common.utils.data_utils import download_and_extract_data, pd_train_test_split, cal_md5

url = "/"

def prepare_file():
    with open("/tmp/xfl/dataset/unit_test/test_raw.txt", "w") as f:
        f.write("unit_test")

@pytest.fixture(scope="module", autouse=True)
def env():
    if not os.path.exists("/tmp/xfl/dataset/unit_test"):
        os.makedirs("/tmp/xfl/dataset/unit_test")
    prepare_file()
    yield
    if os.path.exists("/tmp/xfl/dataset/unit_test"):
        shutil.rmtree("/tmp/xfl/dataset/unit_test")

@pytest.mark.parametrize('ftype, dst_file, data_folder', [("gz", "/tmp/xfl/dataset/unit_test/test.txt.gz", None),("gz", "/tmp/xfl/dataset/unit_test/test.txt.gz","/tmp/xfl/dataset/unit_test/")])
def test_download_and_extract_data(httpserver, ftype, dst_file, data_folder):
    src_file = None
    if ftype == "gz":
        src_file = "/tmp/xfl/dataset/unit_test/test_raw.txt.gz"
        f_ungz = open("/tmp/xfl/dataset/unit_test/test_raw.txt",'rb') 
        f_gz = gzip.open(src_file,'wb') 
        f_gz.writelines(f_ungz) 
        f_ungz.close()
        f_gz.close()
        to_path = None


    content_f = open(src_file,"rb")
    content = content_f.read()
    httpserver.expect_request(url).respond_with_data(content)
    download_and_extract_data(httpserver.url_for(url), None, dst_file, data_folder=data_folder, to_path=to_path)

    if ftype == "gz":
        with open("/tmp/xfl/dataset/unit_test/test.txt","r") as f: 
            assert f.readline() == "unit_test"

def test_cal_md5():
    md5 = cal_md5("/tmp/xfl/dataset/unit_test/test_raw.txt")
    assert md5 == "d16f7309f3bfab471bad7a55b919f044"


def test_pd_train_test_split():
    case_df = pd.DataFrame({
        'x0': np.arange(100),
        'x1': np.arange(100),
        'x2': 2 * np.arange(100) - 40.0,
        'x3': 3 * np.arange(100) + 1.0,
        'x4': np.arange(100)[::-1]
    })

    train_df, test_df = pd_train_test_split(case_df, 0.3)
    assert len(train_df) == 70
    assert len(test_df) == 30