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

import service.fed_config
from algorithm.framework.vertical.sampler.label_trainer import VerticalSamplerLabelTrainer
from algorithm.framework.vertical.sampler.trainer import VerticalSamplerTrainer
from common.communication.gRPC.python.channel import BroadcastChannel
from common.communication.gRPC.python.commu import Commu


@pytest.fixture(scope="module", autouse=True)
def env():
    # 准备目录
    Commu.node_id = "node-1"
    Commu.trainer_ids = ['node-1', 'node-2']
    Commu.scheduler_id = 'assist_trainer'
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    # 测试用例
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': [0] * 1000,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = np.where(
        case_df['x0'] + case_df['x2'] + case_df['x3'] > 2.5, 1, 0)
    case_df[['y', 'x0', 'x1', 'x2']].to_csv(
        "/opt/dataset/unit_test/guest.csv", index=True, index_label='id')
    case_df[['x3', 'x4']].to_csv(
        "/opt/dataset/unit_test/host.csv", index=True, index_label='id')
    yield
    # 清除测试数据
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")


@pytest.fixture()
def get_label_trainer_conf():
    with open("python/algorithm/config/vertical_sampler/label_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["dataset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["dataset"][0]["name"] = "guest.csv"
        conf["input"]["dataset"][0]["has_id"] = True
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/vertical_sampler/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["dataset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["dataset"][0]["name"] = "host.csv"
        conf["input"]["dataset"][0]["has_id"] = True
        conf["output"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


class TestVerticalSampler:
    @pytest.mark.parametrize('datatype, fraction', [("csv", {"percentage": 1}), ("json", {"percentage": 1}),
                                                    ("csv", {"wrong_key": 1})])
    def test_label_trainer_default(self, get_label_trainer_conf, datatype, fraction, mocker):
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )
        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )

        conf = copy.deepcopy(get_label_trainer_conf)
        conf["train_info"]["train_params"]["marketing_specified"] = {}
        conf["input"]["dataset"][0]["type"] = datatype
        conf["train_info"]["train_params"]["fraction"] = fraction
        if datatype == "csv":
            if fraction == {"percentage": 1}:
                ls = VerticalSamplerLabelTrainer(conf)
                assert len(ls.data) == 1000
                ls.fit()
                assert os.path.exists(
                    "/opt/checkpoints/unit_test/temp/sampled_data_[STAGE_ID].csv")
                if ls.save_id:
                    assert os.path.exists(
                        "/opt/checkpoints/unit_test/temp/sampled_id_[STAGE_ID].json")
            elif fraction == {"wrong_key": 1}:
                with pytest.raises(NotImplementedError) as e:
                    ls = VerticalSamplerLabelTrainer(conf)
                    ls.fit()
                    exec_msg = e.value.args[0]
                    assert exec_msg == "Fraction key {} is not supported.".format(
                        list(ls.fraction.keys())[0])
        else:
            with pytest.raises(NotImplementedError) as e:
                ls = VerticalSamplerLabelTrainer(conf)
                exec_msg = e.value.args[0]
                assert exec_msg == "Dataset type {} is not supported.".format(
                    ls.input["dataset"][0]["type"])
        # test more than one data config
        conf1 = copy.deepcopy(get_label_trainer_conf)
        conf1["input"]["dataset"] = [{}, {}]
        try:
            VerticalSamplerLabelTrainer(conf1)
        except Exception:
            pass
        # test no data config
        conf2 = copy.deepcopy(get_label_trainer_conf)
        conf2["input"]["dataset"] = []
        with pytest.raises(NotImplementedError) as e:
            ls = VerticalSamplerLabelTrainer(conf2)
            exec_msg = e.value.args[0]
            assert exec_msg == "Dataset was not configured."

    @pytest.mark.parametrize('method, strategy, fraction, infer_params', [
        ("random", "downsample", {"percentage": 0.1}, {
         "threshold_method": "percentage", "threshold": 0.1}),
        ("random", "downsample", {"number": 10}, {
        }), ("random", "upsample", {"percentage": 1.1}, {}),
        ("random", "downsample", {"percentage": 1.2}, {
        }), ("random", "upsample", {"percentage": -0.1}, {}),
        ("random", "sample", {"percentage": 0.1}, {}), ("stratify", "downsample",
                                                        {"labeled_percentage": [[0, 0.1], [1, 0.2]]}, {}),
        ("stratify", "upsample", {
         "labeled_percentage": [[0, 1.1], [1, 1.2]]}, {}),
        ("stratify", "sample", {"percentage": 0.1}, {}),
        ("try", "downsample", {"percentage": 0.1}, {})
    ])
    def test_label_trainer_fit(self, get_label_trainer_conf, method, strategy, fraction, mocker, infer_params):
        mocker.patch.object(
            BroadcastChannel, "broadcast", return_value=0
        )
        conf = copy.deepcopy(get_label_trainer_conf)
        conf["train_info"]["train_params"]["method"] = method
        conf["train_info"]["train_params"]["strategy"] = strategy
        conf["train_info"]["train_params"]["fraction"] = fraction
        conf["train_info"]["train_params"]["marketing_specified"] = infer_params

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )

        ls = VerticalSamplerLabelTrainer(conf)
        if method == "random" and infer_params == {}:
            if strategy == 'downsample' and fraction == {"percentage": 0.1}:
                ls.fit()
                assert len(ls.sample_ids) == int(fraction * 1000)
            elif strategy == 'downsample' and fraction == {"number": 10}:
                ls.fit()
                assert len(ls.sample_ids) == 10
            elif strategy == 'upsample' and fraction == {"percentage": 1.1}:
                ls.fit()
                assert len(ls.sample_ids) == int(fraction["percentage"] * 1000)
                assert len(set(ls.sample_ids)) < len(ls.sample_ids)
            elif strategy == 'downsample' and {"percentage": 1.2}:
                with pytest.raises(ValueError) as e:
                    ls.fit()
                    exec_msg = e.value.args[0]
                    assert exec_msg == "Fraction should be a numeric number between 0 and 1"
            elif strategy == 'upsample' and fraction == {"percentage": -0.1}:
                with pytest.raises(ValueError) as e:
                    ls.fit()
                    exec_msg = e.value.args[0]
                    assert exec_msg == "Fraction should be a numeric number larger than 0"
            else:
                with pytest.raises(NotImplementedError) as e:
                    ls.fit()
                    exec_msg = e.value.args[0]
                    assert exec_msg == "Strategy type {} is not supported.".format(
                        ls.strategy)
        elif method == "random" and infer_params != {}:
            ls.data = ls.data[["y"]]
            ls.fit()
            assert len(ls.sample_ids) == int(len(ls.data) * ls.threshold)
            conf1 = copy.deepcopy(conf)
            conf1["train_info"]["train_params"]["marketing_specified"]["threshold_method"] = "number"
            conf1["train_info"]["train_params"]["marketing_specified"]["threshold"] = 100
            ls1 = VerticalSamplerLabelTrainer(conf1)
            ls1.data = ls1.data[["y"]]
            ls1.fit()
            assert len(ls1.sample_ids) == 100
            conf1_1 = copy.deepcopy(conf1)
            conf1_1["train_info"]["train_params"]["marketing_specified"]["threshold"] = 10000
            ls1_1 = VerticalSamplerLabelTrainer(conf1_1)
            ls1_1.data = ls1_1.data[["y"]]
            with pytest.raises(OverflowError) as e:
                ls1_1.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == "Threshold number {} is larger than input data size.".format(
                    ls1_1.threshold)
            conf2 = copy.deepcopy(conf)
            conf2["train_info"]["train_params"]["marketing_specified"]["threshold_method"] = "score"
            conf2["train_info"]["train_params"]["marketing_specified"]["threshold"] = 0.5
            ls2 = VerticalSamplerLabelTrainer(conf2)
            ls2.data = ls2.data[["y"]]
            ls2.fit()
            assert len(ls2.sample_ids) == np.sum(ls2.data)[0]
        elif method == "stratify" and infer_params == {}:
            if strategy == 'downsample' and fraction == {"labeled_percentage": [[0, 0.1], [1, 0.2]]}:
                ls.fit()
                assert len(ls.sample_ids) == int(
                    ls.label_count[0] * 0.1) + int(ls.label_count[1] * 0.2)
            elif strategy == 'upsample' and fraction == {"labeled_percentage": [[0, 1.1], [1, 1.2]]}:
                ls.fit()
                assert len(ls.sample_ids) == int(
                    ls.label_count[0] * 1.1) + int(ls.label_count[1] * 1.2)
                assert len(set(ls.sample_ids)) < len(ls.sample_ids)
            else:
                with pytest.raises(NotImplementedError) as e:
                    ls.fit()
                    exec_msg = e.value.args[0]
                    assert exec_msg == "Strategy type {} is not supported.".format(
                        ls.strategy)
        else:
            with pytest.raises(NotImplementedError) as e:
                ls.fit()
                exec_msg = e.value.args[0]
                assert exec_msg == "Method type {} is not supported.".format(
                    ls.method)

    def test_trainer(self, get_label_trainer_conf, mocker):
        conf = copy.deepcopy(get_label_trainer_conf)

        mocker.patch.object(
            service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
        )

        mocker.patch.object(
            service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
        )

        ls = VerticalSamplerTrainer(conf)

        def mock_recv(*args, **kwargs):
            num = int(conf["train_info"]["train_params"]
                      ["fraction"]["percentage"] * len(ls.data))
            return ls.data.index[:num]

        mocker.patch.object(
            BroadcastChannel, "recv", side_effect=mock_recv
        )
        ls.fit()
