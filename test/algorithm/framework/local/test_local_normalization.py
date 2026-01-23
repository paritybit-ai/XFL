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
from scipy.linalg import norm
from google.protobuf import json_format

from algorithm.framework.local.normalization.label_trainer import \
	LocalNormalizationLabelTrainer as LocalNormalization
from common.model.python.feature_model_pb2 import NormalizationModel


@pytest.fixture(scope="module", autouse=True)
def env():
	# 准备目录
	if not os.path.exists("/tmp/xfl/dataset/unit_test"):
		os.makedirs("/tmp/xfl/dataset/unit_test")
	if not os.path.exists("/tmp/xfl/checkpoints/unit_test"):
		os.makedirs("/tmp/xfl/checkpoints/unit_test")
	# 测试用例
	case_df = pd.DataFrame({
		'x0': np.random.random(1000),
		'x1': [0] * 1000,
		'x2': 2 * np.random.random(1000) + 1.0
	})
	case_df['y'] = np.where(case_df['x0'] + case_df['x2'] > 2.5, 1, 0)
	case_df[['y', 'x0', 'x1', 'x2']].head(800).to_csv(
		"/tmp/xfl/dataset/unit_test/train.csv", index=True, index_label='id'
	)
	case_df[['y', 'x0', 'x1', 'x2']].tail(200).to_csv(
		"/tmp/xfl/dataset/unit_test/test.csv", index=True, index_label="id"
	)
	yield
	# 清除测试数据
	if os.path.exists("/tmp/xfl/dataset/unit_test"):
		shutil.rmtree("/tmp/xfl/dataset/unit_test")
	if os.path.exists("/tmp/xfl/checkpoints/unit_test"):
		shutil.rmtree("/tmp/xfl/checkpoints/unit_test")


@pytest.fixture()
def get_conf():
	with open("python/algorithm/config/local_normalization/label_trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train.csv"
		conf["input"]["valset"][0]["path"] = "/tmp/xfl/dataset/unit_test"
		conf["input"]["valset"][0]["name"] = "test.csv"
		conf["output"]["path"] = "/tmp/xfl/checkpoints/unit_test"
		conf["output"]["trainset"]["name"] = "normalized_train.csv"
		conf["output"]["valset"]["name"] = "normalized_test.csv"
	yield conf


class TestLocalNormalization:
	def test_default(self, get_conf, mocker):
		mocker.patch("service.fed_control._send_progress")
		ln = LocalNormalization(get_conf)
		assert len(ln.train_data) == 800
		assert len(ln.valid_data) == 200
		ln.fit()
		assert os.path.exists("/tmp/xfl/checkpoints/unit_test/normalized_train.csv")
		assert os.path.exists("/tmp/xfl/checkpoints/unit_test/normalized_test.csv")
		assert len(pd.read_csv("/tmp/xfl/checkpoints/unit_test/normalized_train.csv")) == 800
		assert len(pd.read_csv("/tmp/xfl/checkpoints/unit_test/normalized_test.csv")) == 200

	@pytest.mark.parametrize('axis, norm_', [
		(1, 'l1'), (1, 'l2'), (1, 'max'), (1, 'other'), (0, 'l1'), (0, 'l2'), (0, 'max'), (0, 'other2'), (2, 'l1')
	])
	def test_fit(self, get_conf, axis, norm_, mocker):
		conf = copy.deepcopy(get_conf)
		conf["train_info"]["train_params"]["axis"] = axis
		conf["train_info"]["train_params"]["norm"] = norm_
		mocker.patch("service.fed_control._send_progress")
		ln = LocalNormalization(conf)
		check_output = True
		if axis == 0:
			if norm_ == 'l1':
				ln.fit()
				assert (np.abs(ln.train_data[['x1']].apply(lambda x: norm(x, ord=1)) - 0.0) < 1e-6).all()
			elif norm_ == 'l2':
				ln.fit()
				assert (np.abs(ln.train_data[['x1']].apply(lambda x: norm(x, ord=2)) - 0.0) < 1e-6).all()
				assert (np.abs(ln.train_data[['x0', 'x2']].apply(lambda x: norm(x, ord=2)) - 1.0) < 1e-6).all()
			elif norm_ == 'max':
				ln.fit()
				assert (np.abs(ln.train_data[['x1']].apply(lambda x: np.max(np.abs(x))) - 0.0) < 1e-6).all()
				assert (ln.train_data[['x0', 'x2']].apply(lambda x: np.max(np.abs(x))).to_numpy() <= 1.0).all()
			else:
				with pytest.raises(NotImplementedError) as e:
					ln.fit()
					exec_msg = e.value.args[0]
					assert exec_msg == "norm {} is invalid.".format(norm_)
				check_output = False
		elif axis == 1:
			if norm_ == 'l1':
				ln.fit()
				assert (abs(
					ln.train_data[['x1', 'x2', 'x0']].apply(lambda x: norm(x, ord=1),
															axis=1).to_numpy() - 1.0) < 1e-6).all()
			elif norm_ == 'l2':
				ln.fit()
				assert (abs(
					ln.train_data[['x1', 'x2', 'x0']].apply(lambda x: norm(x, ord=2),
															axis=1).to_numpy() - 1.0) < 1e-6).all()
			elif norm_ == 'max':
				ln.fit()
				assert (ln.train_data[['x1', 'x2', 'x0']].apply(lambda x: np.max(np.abs(x)),
																axis=1).to_numpy() <= 1.0).all()
			else:
				with pytest.raises(NotImplementedError) as e:
					ln.fit()
					exec_msg = e.value.args[0]
					assert exec_msg == "norm {} is invalid.".format(norm_)
				check_output = False
		else:
			with pytest.raises(ValueError) as e:
				ln.fit()
				exec_msg = e.value.args[0]
				assert exec_msg == "axis {} is invalid.".format(axis)
			check_output = False
		if check_output:
			with open(conf["output"]["path"] + '/' + conf["output"]["proto_model"]["name"], 'rb') as f:
				byte_str = f.read()
			m = NormalizationModel()
			m.ParseFromString(byte_str)
			d = json_format.MessageToDict(m,
			                              preserving_proto_field_name=True)

			assert d.get("axis") == axis
			if axis == 0:
				assert len(d.get("normalizer")) == 3
			elif axis == 1:
				assert d.get("norm") == norm_

	@pytest.mark.parametrize('feature_name', ['x0', 'myf'])
	def test_feature_wise(self, get_conf, feature_name, mocker):
		conf = copy.deepcopy(get_conf)
		conf["train_info"]["train_params"]["norm"] = 'l2'
		conf["train_info"]["train_params"]["feature_norm"] = {feature_name: {"norm": 'l1'}}
		mocker.patch("service.fed_control._send_progress")
		ln = LocalNormalization(conf)

		if feature_name in ln.train_data.columns:
			ln.fit()
			assert np.abs(norm(ln.train_data['x0'].to_numpy(), ord=1) - 1.0) < 1e-6
			assert np.abs(norm(ln.train_data['x0'].to_numpy(), ord=2) - 1.0) >= 1e-6
			assert np.abs(norm(ln.train_data['x2'].to_numpy(), ord=2) - 1.0) < 1e-6
		else:
			with pytest.raises(KeyError):
				ln.fit()
