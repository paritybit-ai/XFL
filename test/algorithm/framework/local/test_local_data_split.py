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


import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithm.framework.local.data_split.label_trainer import \
    LocalDataSplitLabelTrainer as LocalDataSplit


@pytest.fixture(scope="module", autouse=True)
def env():
    #
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    #
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': [0] * 1000,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = np.where(case_df['x1'] + case_df['x2'] > 2.5, 1, 0)
    case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']].to_csv("/opt/dataset/unit_test/dataset.csv", index=True,
                                                        index_label='id')
    case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']].to_csv("/opt/dataset/unit_test/dataset_opt.csv", index=True,
                                                        index_label='id')
    yield
    #
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")


@pytest.fixture()
def get_conf():
    with open("python/algorithm/config/local_data_split/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["dataset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["dataset"][0]["name"] = "dataset.csv"
        conf["output"]["trainset"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["trainset"]["name"] = "data_train.csv"
        conf["output"]["valset"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["valset"]["name"] = "data_test.csv"
    yield conf


class TestLocalDataSplit:
    @pytest.mark.parametrize('dataset_name', ["dataset.csv", None])
    def test_default(self, get_conf, dataset_name):
        conf = copy.deepcopy(get_conf)
        conf["input"]["dataset"][0]["name"] = dataset_name

        if dataset_name == "dataset.csv":
            lds = LocalDataSplit(conf)
            assert lds.files == ["/opt/dataset/unit_test/dataset.csv"]
        else:
            lds = LocalDataSplit(conf)
            assert len(lds.files) == 2
        #
        conf1 = copy.deepcopy(get_conf)
        conf1["input"]["dataset"] = []
        with pytest.raises(NotImplementedError) as ee:
            lds = LocalDataSplit(conf1)
            exec_msg = ee.value.args[0]
            assert exec_msg == "Dataset was not configured."

    @pytest.mark.parametrize('dataset_name, shuffle_params, header, batchSize',
                             [("dataset.csv", True, True, 1000),
                              ("dataset.csv", True, False, 1000),
                              ("dataset.csv", False, False, 1000),
                              ("dataset.csv", False, True, 1000),
                              (None, True, True, 1000),
                              (None, True, False, 1000),
                              (None, False, False, 1000),
                              (None, False, True, 1000)])
    def test_fit(self, get_conf, shuffle_params, dataset_name, header, batchSize):
        conf = copy.deepcopy(get_conf)
        conf["train_info"]["params"]["shuffle_params"] = shuffle_params
        conf["input"]["dataset"][0]["name"] = dataset_name
        conf["input"]["dataset"][0]["header"] = header
        conf["train_info"]["params"]["batch_size"] = batchSize
        output_train = Path(conf["output"]["trainset"]["path"], conf["output"]["trainset"]["name"])
        output_val = Path(conf["output"]["valset"]["path"], conf["output"]["valset"]["name"])
        lds = LocalDataSplit(conf)
        if dataset_name == "dataset.csv":
            if not shuffle_params:
                if header:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 801 and len(val) == 201
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] == "id"
                    assert train.iloc[1, 0] == '0' and val.iloc[1, 0] == '800'
                else:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 800 and len(val) == 201
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] != "id"
                    assert train.iloc[1, 0] == '0' and val.iloc[0, 0] == 799
            else:
                if header:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 801 and len(val) == 201
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] == "id"
                    assert train.iloc[1, 0] != '0' and val.iloc[1, 0] != '800'
                else:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 800 and len(val) == 201
                    assert train.iloc[0, 0] != "id" and val.iloc[0, 0] != "id"
                    assert train.iloc[1, 0] != '0' and val.iloc[0, 0] != '799'
        else:
            if not shuffle_params:
                if header:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 1601 and len(val) == 401
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] == "id"
                    assert train.iloc[1, 0] == '0' and val.iloc[1, 0] == '600'
                else:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 1601 and len(val) == 401
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] != "id"
                    assert train.iloc[1, 0] == '0' and val.iloc[0, 0] == 599
            else:
                if header:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 1601 and len(val) == 401
                    assert train.iloc[0, 0] == "id" and val.iloc[0, 0] == "id"
                    assert train.iloc[1, 0] != '0' and val.iloc[1, 0] != '600'
                else:
                    lds.fit()
                    train = pd.read_csv(output_train, header=None)
                    val = pd.read_csv(output_val, header=None)
                    assert len(train) == 1601 and len(val) == 401
                    assert train.iloc[0, 0] != "id" and val.iloc[0, 0] != "id"
                    assert train.iloc[1, 0] != '0' and val.iloc[0, 0] != '599'
