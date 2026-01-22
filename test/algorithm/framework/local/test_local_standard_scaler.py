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

from algorithm.framework.local.standard_scaler.label_trainer import \
	LocalStandardScalerLabelTrainer as LocalStandardScaler


@pytest.fixture(scope="module", autouse=True)
def env():
	# 准备目录
	if not os.path.exists("/tmp/xfl/dataset/unit_test"):
		os.makedirs("/tmp/xfl/dataset/unit_test")
	if not os.path.exists("/tmp/xfl/checkpoints"):
		os.makedirs("/tmp/xfl/checkpoints")
	# 测试用例
	case_df = pd.DataFrame({
		'x0': np.random.random(1000),
		'x1': [0] * 1000,
		'x2': 2 * np.random.random(1000) + 1.0
	})
	case_df['y'] = np.where(case_df['x0'] + case_df['x2'] > 2.5, 1, 0)
	case_df[['y', 'x0', 'x1', 'x2']].head(800).to_csv(
		"/tmp/xfl/dataset/unit_test/train.csv", index=True
	)
	case_df[['y', 'x0', 'x1', 'x2']].tail(200).to_csv(
		"/tmp/xfl/dataset/unit_test/test.csv", index=True
	)
	yield
	# 清除测试数据
	if os.path.exists("/tmp/xfl/dataset/unit_test"):
		shutil.rmtree("/tmp/xfl/dataset/unit_test")


@pytest.fixture()
def get_conf():
	with open("python/algorithm/config/local_normalization/label_trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train.csv"
		conf["input"]["valset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["input"]["valset"][0]["name"] = "test.csv"
		conf["output"]["trainset"]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["output"]["trainset"]["name"] = "train.csv"
		conf["output"]["valset"]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["output"]["valset"]["name"] = "test.csv"
	yield conf


class TestLocalStandardScaler:

	def test_init(self, get_conf):
		ln = LocalStandardScaler(get_conf)
		assert len(ln.train_data) == 800
		assert len(ln.valid_data) == 200

	@pytest.mark.parametrize('with_mean, with_std', [
		(True, True), (True, False), (False, True), (False, False)
	])
	def test_fit(self, get_conf, with_mean, with_std, mocker):
		conf = copy.deepcopy(get_conf)
		conf["train_info"]["train_params"]["with_mean"] = with_mean
		conf["train_info"]["train_params"]["with_std"] = with_std
		mocker.patch("service.fed_control._send_progress")
		ln = LocalStandardScaler(conf)
		ln.fit()
		if with_mean:
			assert (np.abs(ln.train_data[['x0', 'x1', 'x2']].mean()) < 1e-6).all()
		if with_std:
			assert (np.abs(ln.train_data[['x0', 'x2']].std() - 1.0) < 1e-6).all()

	@pytest.mark.parametrize('feature_name', ['x0', 'myf'])
	def test_feature_wise(self, get_conf, feature_name, mocker):
		conf = copy.deepcopy(get_conf)
		conf["train_info"]["train_params"]["with_mean"] = False
		conf["train_info"]["train_params"]["with_std"] = False
		conf["train_info"]["train_params"]["feature_standard"] = {
			feature_name: {"with_mean": True, "with_std": True}
		}
		mocker.patch("service.fed_control._send_progress")
		ln = LocalStandardScaler(conf)

		if feature_name in ln.train_data.columns:
			ln.fit()
			assert (np.abs(ln.train_data['x0'].mean()) < 1e-6).all()
			assert (np.abs(ln.train_data['x2'].mean()) > 1e-6).all()
		else:
			with pytest.raises(KeyError):
				ln.fit()
