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


import os
import shutil

import pytest

from algorithm.core.output import TableSaver


@pytest.fixture(scope="module", autouse=True)
def env():
    if not os.path.exists("/tmp/xfl/dataset/unit_test"):
        os.makedirs("/tmp/xfl/dataset/unit_test")
    yield
    if os.path.exists("/tmp/xfl/dataset/unit_test"):
        shutil.rmtree("/tmp/xfl/dataset/unit_test")

class TestTableSaver():
    def test_save(self):
        ts = TableSaver('/tmp/xfl/dataset/unit_test/table.tb')
        ts.save(0,{"x0":1.0,"x1":2.0},prefix='unit',suffix="test", append=False)
        assert os.path.exists(
            "/tmp/xfl/dataset/unit_test/unit_table_test.tb")

        ts.save(1,{"x2":3.0,"x3":4.0},prefix='unit',suffix="test", append=True)
        with open("/tmp/xfl/dataset/unit_test/unit_table_test.tb",'r') as f:
                assert f.readlines() == ['epoch,x0,x1\n', '0,1,2\n', '1,3,4\n']
