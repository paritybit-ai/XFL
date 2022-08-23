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

import numpy as np
import pandas as pd
import pytest

from algorithm.framework.local.feature_preprocess.label_trainer import \
    LocalFeaturePreprocessLabelTrainer as LocalPreprocess


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
    case_df[['y', 'x00', 'x01', 'x02', 'x03']].head(800).to_csv(
        "/opt/dataset/unit_test/train.csv", index=True, index_label='id'
    )
    case_df[['y', 'x00', 'x01', 'x02', 'x03']].tail(200).to_csv(
        "/opt/dataset/unit_test/test.csv", index=True, index_label="id"
    )
    yield
    #
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")


@pytest.fixture()
def get_conf():
    with open("python/algorithm/config/local_feature_preprocess/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train.csv"
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test.csv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["trainset"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["trainset"]["name"] = "preprocessed_train.csv"
        conf["output"]["valset"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["valset"]["name"] = "preprocessed_test.csv"
    yield conf


class TestLocalNormalization:
    @pytest.mark.parametrize('datatype', ["csv", "json"])
    def test_default(self, get_conf, datatype):
        conf = copy.deepcopy(get_conf)
        conf["input"]["trainset"][0]["type"] = datatype

        if datatype == "csv":
            lp = LocalPreprocess(conf)
            assert len(lp.train) == 800
            assert len(lp.val) == 200
        else:
            with pytest.raises(NotImplementedError) as ee:
                lp = LocalPreprocess(conf)
                exec_msg = ee.value.args[0]
                assert exec_msg == "Dataset type {} is not supported.".format(lp.input["trainset"]["type"])
        #
        conf1 = copy.deepcopy(get_conf)
        conf1["input"]["trainset"] = [{}, {}]
        try:
            LocalPreprocess(conf1)
        except:
            pass
        #
        conf2 = copy.deepcopy(get_conf)
        conf2["input"]["trainset"] = []
        with pytest.raises(NotImplementedError) as ee:
            lp = LocalPreprocess(conf2)
            exec_msg = ee.value.args[0]
            assert exec_msg == "Trainset was not configured."

    @pytest.mark.parametrize('missing_params, outlier_params', [
        ({}, {"outlier_feat_params": {"x03": {"outlier_values": 999}, "x01": {}},
              "outlier_values": ["", " ", "nan", "none", "null", "na", "None"]}),
        ({}, {"outlier_feat_params": {"x03": {"outlier_values": 999}, "x01": {}}}),
        ({"fill_value": None, "missing_feat_params":
            {"x01": {"fill_value": None, "missing_values": None, "strategy": "median"}, "x00": {}},
          "missing_values": None, "strategy": "mean"}, {}),
        ({"fill_value": None, "missing_feat_params": {}, "missing_values": None, "strategy": "mean"},
         {"outlier_feat_params": {}, "outlier_values": 999}),
        ({"fill_value": None, "missing_feat_params": {"x01": {"fill_value": 1, "missing_values": 999,
                                                              "strategy": "constant"}, "x00": {}},
          "missing_values": None, "strategy": "mean"},
         {"outlier_feat_params": {"x03": {"outlier_values": 999}, "x01": {}}, "outlier_values": 999}),
        ({}, {}),
        ({"fill_value": 1, "missing_values": 'nan', "strategy": "constant"},
         {"outlier_feat_params": {"x03": {"outlier_values": 999}, "x01": {}}, "outlier_values": [999, -999]}),
        ({"fill_value": 1, "missing_values": 'nan', "strategy": "constant"},
         {"outlier_feat_params": {"x03": {"outlier_values": 999}, "x01": {}}, "outlier_values": 999})
    ])
    def test_fit(self, get_conf, missing_params, outlier_params):
        conf = copy.deepcopy(get_conf)
        conf["train_info"]["params"]["missing_params"] = missing_params
        conf["train_info"]["params"]["outlier_params"] = outlier_params
        lp = LocalPreprocess(conf)
        if missing_params == {}:
            if outlier_params == {}:
                assert lp.imputer_values_overall == []
                assert lp.impute_dict == {}
                lp.fit()
                assert np.sum(lp.train["x00"].isna()) > 0
            else:
                if lp.outlier_values_overall:
                    assert len(set(lp.imputer_values_overall).difference({np.NaN, '', None, ' ', 'nan', 'none', 'null',
                                                                          'na', 'None', 999})) == 0
                    assert len(set(lp.impute_dict["x03"]["missing_values"]).difference(
                        {np.NaN, '', None, ' ', 'nan', 'none', 'null', 'na', 'None', 999})) == 0
                    lp.fit()
                    assert np.sum(lp.train["x00"].isna()) == 0
                else:
                    assert len(lp.impute_dict) == 1
                    lp.fit()
                    assert np.sum(lp.train["x00"].isna()) > 0
        else:
            if outlier_params == {}:
                assert lp.outlier_conf == {}
                lp.impute_dict["x01"]["strategy"] = "median"
            else:
                if lp.missing_feat_conf == {} and lp.outlier_feat_conf == {}:
                    assert not lp.feature_flag
                    assert len(lp.impute_dict) == len(lp.train.columns)
                else:
                    assert lp.feature_flag
            lp.fit()
            assert np.sum(lp.train["x00"].isna()) == 0
        assert len(set(lp.train.columns).difference({'y', 'x00', 'x01', 'x03', 'x02_0', 'x02_1'})) == 0
        # test no onehot
        conf1 = copy.deepcopy(get_conf)
        conf1["train_info"]["params"]["onehot_params"] = {}
        lp = LocalPreprocess(conf1)
        lp.fit()
        assert len(set(lp.train.columns).difference({'y', 'x00', 'x01', 'x03', 'x02'})) == 0
