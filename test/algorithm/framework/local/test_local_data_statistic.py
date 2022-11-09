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

from algorithm.framework.local.data_statistic.label_trainer import \
    LocalDataStatisticLabelTrainer as LocalDataStatistic
from algorithm.framework.local.data_statistic.trainer import \
    LocalDataStatisticTrainer as LocalDataStatisticTrainer


@pytest.fixture(scope="module", autouse=True)
def env():
    #
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    #
    case_df = pd.DataFrame({
        'x01': np.random.random(1000),
        'x00': [np.NaN, '', None, ' ', 'nan'] + [0] * 995,
        'x03': 2 * np.random.random(1000) + 1.0,
        'x02': [0] * 300 + [1] * 700
    })
    case_df['y'] = np.where(case_df['x01'] + case_df['x02'] > 2.5, 1, 0)
    case_df[['y', 'x00', 'x01', 'x02', 'x03']].to_csv(
        "/opt/dataset/unit_test/data.csv", index=True, index_label='id'
    )
    case_df[['y', 'x00', 'x01', 'x02', 'x03']].to_csv(
        "/opt/dataset/unit_test/data_opt.csv", index=True, index_label='id'
    )

    yield
    #
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test_1"):
        shutil.rmtree("/opt/checkpoints/unit_test_1")


@pytest.fixture()
def get_conf():
    with open("python/algorithm/config/local_data_statistic/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["dataset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["dataset"][0]["name"] = "data.csv"
        conf["output"]["path"] = "/opt/checkpoints/unit_test_1"
    yield conf


class TestLocalDataStatistic:
    @pytest.mark.parametrize('dataset', [[
        {
            "type": "csv",
            "path": "/opt/dataset/unit_test",
            "name": "data.csv",
            "has_label": True,
            "has_id": True
        }
    ],
        [
            {
                "type": "csv",
                "path": "/opt/dataset/unit_test",
                "name": "data.csv",
                "has_label": True,
                "has_id": True
            },
            {
                "type": "csv",
                "path": "/opt/dataset/unit_test",
                "name": "data_opt.csv",
                "has_label": True,
                "has_id": True
            }
        ]
    ])
    def test_default(self, get_conf, dataset):
        conf = copy.deepcopy(get_conf)
        conf["input"]["dataset"] = dataset
        lds = LocalDataStatistic(conf)

        if len(dataset) == 1:
            assert len(lds.data) == 1000
        else:
            assert len(lds.data) == 2000

    @pytest.mark.parametrize('train_info_params', [{}, {"quantile": [0.5, 0.8, 0.9]}])
    def test_fit(self, get_conf, train_info_params):
        conf = copy.deepcopy(get_conf)
        conf["train_info"]["train_params"] = train_info_params
        lds = LocalDataStatistic(conf)
        if train_info_params == {}:
            assert lds.quantile == [0.25, 0.5, 0.75]
            # test for no label
            conf1 = copy.deepcopy(get_conf)
            conf1["input"]["dataset"][0]["has_label"] = False
            lds = LocalDataStatistic(conf1)
            lds.fit()
            assert "label_num" not in lds.summary_dict.keys()
            assert lds.summary_dict["row_num"] == 1000
            assert lds.summary_dict["column_num"] == 5
            assert lds.summary_dict["feature_names"] == ["y", "x00", "x01", "x02", "x03"]
            assert lds.summary_dict["missing_ratio"]["x00"] == float("%.6f" % (5/1000))
        else:
            assert lds.quantile == [0.5, 0.8, 0.9]
            lds.fit()
            assert len(set(lds.summary_dict.keys()).difference(
                {"mean", "median", "missing_ratio", "min", "max", "variance", "std", "quantile", "skewness", "kurtosis",
                 "quantile", "row_num", "label_num", "column_num", "feature_names"})) == 0
            assert lds.summary_dict["column_num"] == 4
            assert lds.summary_dict["feature_names"] == ["x00", "x01", "x02", "x03"]

    def test_trainer(self, get_conf):
        LocalDataStatisticTrainer(get_conf)



