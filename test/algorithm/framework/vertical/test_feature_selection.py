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
import json
import pytest
import pickle

import pandas as pd
import numpy as np

import service.fed_config
from common.communication.gRPC.python.channel import DualChannel, BroadcastChannel
from algorithm.framework.vertical.feature_selection.label_trainer import VerticalFeatureSelectionLabelTrainer
from algorithm.framework.vertical.feature_selection.trainer import VerticalFeatureSelectionTrainer
from common.communication.gRPC.python.commu import Commu

def prepare_data():
	# iv output
	d = {
		"iv": {
			'x0': 0.35,
			'x1': 0.8,
			'x2': 0.09,
			'ZY8HUnfdEtpr6LIW': 0.1,
			'McqB4XwT3NSoWEyt': 0.5
		}
	}
	with open("/opt/checkpoints/unit_test/vertical_binning_woe_iv.json", 'w') as f:
		json.dump(d, f)

	d = {
		"feature_mapping": {
			'x3': 'ZY8HUnfdEtpr6LIW',
			'x4': 'McqB4XwT3NSoWEyt'
		}
	}
	with open("/opt/checkpoints/unit_test/vertical_binning_woe_iv_feature_mapping.json", 'w') as f:
		json.dump(d, f)

	# pearson output
	c = {
		"corr": [[1.0, 0.0, 0.0, 0.0, 0.8],
				 [0.0, 1.0, 0.0, 0.0, 0.0],
				 [0.0, 0.0, 1.0, 0.0, 0.0],
				 [0.0, 0.0, 0.0, 1.0, 0.0],
				 [0.8, 0.0, 0.0, 0.0, 1.0]],
		"features": ["x0", "x1", "x2", "sIcVutHjKxZyqUg1", "FkYzRlp1ySwdnm0N"],
		"feature_source": ["node-1", "node-1", "node-1", "node-2", "node-2"]
	}
	with open("/opt/checkpoints/unit_test/pearson_guest.pkl", 'wb') as f:
		pickle.dump(c, f)

	c = {
		"corr": [[1.0, 0.0],
				 [0.0, 1.0]],
		"features": ["x3", "x4"],
		"feature_source": ["node-1", "node-1", "node-1", "node-2", "node-2"],
		"feature_mapping": {
			"x3": "sIcVutHjKxZyqUg1",
			"x4": "FkYzRlp1ySwdnm0N"
		}
	}

	with open("/opt/checkpoints/unit_test/pearson_host.pkl", 'wb') as f:
		pickle.dump(c, f)
	# raw data
	data = pd.DataFrame(
		{
			"y": [1, 1, 0],
			"x0": [1, 2, 3],
			"x1": [1, 1, 1],
			"x2": [2, 2, 2],
			"x3": [2, 3, 1],
			"x4": [2, 2, 2]
		}
	)
	data[["y", "x0", "x1", "x2"]].to_csv(
		"/opt/checkpoints/unit_test/train_guest.csv", header=True, index=True, index_label='id')
	data[["x3", "x4"]].to_csv("/opt/checkpoints/unit_test/train_host.csv", header=True, index=True, index_label='id')

	woe_mapping = {
		"x0": {
			"woe": [-2.550868, -0.386166, 0.382953, 0.69536, 0.385522],
			"binning_split": [26.0, 33.0, 41.0, 50.0, np.inf]
		},
		"x1": {
			"woe": [-0.93796, 0.325627],
			"binning_split": [0.0, np.inf]
		}
	}
	with open("/opt/checkpoints/unit_test/binning_woe_iv_guest.json", 'w') as f:
		json.dump(woe_mapping, f)


@pytest.fixture(scope="module", autouse=True)
def env():
	Commu.node_id = "node-1"
	Commu.trainer_ids = ['node-1','node-2']
	Commu.scheduler_id = "assist_trainer"
	if not os.path.exists("/opt/dataset/unit_test"):
		os.makedirs("/opt/dataset/unit_test")
	if not os.path.exists("/opt/checkpoints/unit_test"):
		os.makedirs("/opt/checkpoints/unit_test")
	prepare_data()
	yield
	if os.path.exists("/opt/dataset/unit_test"):
		shutil.rmtree("/opt/dataset/unit_test")
	if os.path.exists("/opt/checkpoints/unit_test"):
		shutil.rmtree("/opt/checkpoints/unit_test")


@pytest.fixture()
def get_label_trainer_conf():
	with open("python/algorithm/config/vertical_feature_selection/label_trainer.json") as f:
		conf = json.load(f)
		conf["input"]["iv_result"]["path"] = "/opt/checkpoints/unit_test"
		conf["input"]["corr_result"]["path"] = "/opt/checkpoints/unit_test"
		conf["input"]["trainset"] = [
			{
				"type": "csv",
				"path": "/opt/checkpoints/unit_test",
				"name": "train_guest.csv",
				"has_label": True,
				"has_id": True
			}
		]
		conf["input"]["valset"] = [
			{
				"type": "csv",
				"path": "/opt/checkpoints/unit_test",
				"name": "train_guest.csv",
				"has_label": True,
				"has_id": True
			}
		]
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
		conf["output"]["trainset"] = {
			"type": "csv",
			"path": "/opt/checkpoints/unit_test",
			"name": "selected_train_guest.csv",
			"has_label": True,
			"has_id": True
		}
		conf["output"]["valset"] = {
			"type": "csv",
			"path": "/opt/checkpoints/unit_test",
			"name": "selected_train_guest.csv",
			"has_label": True,
			"has_id": True
		}
	yield conf


@pytest.fixture()
def get_trainer_conf():
	with open("python/algorithm/config/vertical_feature_selection/trainer.json") as f:
		conf = json.load(f)
		conf["input"]["iv_result"]["path"] = "/opt/checkpoints/unit_test"
		conf["input"]["corr_result"]["path"] = "/opt/checkpoints/unit_test"
		conf["input"]["trainset"] = [
			{
				"type": "csv",
				"path": "/opt/checkpoints/unit_test",
				"name": "train_host.csv",
				"has_label": False,
				"has_id": True
			}
		]
		conf["input"]["valset"] = [
			{
				"type": "csv",
				"path": "/opt/checkpoints/unit_test",
				"name": "train_host.csv",
				"has_label": False,
				"has_id": True
			}
		]
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
		conf["output"]["trainset"] = {
			"type": "csv",
			"path": "/opt/checkpoints/unit_test",
			"name": "selected_train_host.csv",
			"has_label": False,
			"has_id": True
		}
		conf["output"]["valset"] = {
			"type": "csv",
			"path": "/opt/checkpoints/unit_test",
			"name": "selected_train_host.csv",
			"has_label": False,
			"has_id": True
		}
	yield conf


class TestFeatureSelection:
	@pytest.mark.parametrize("iv, corr, remain", [(0.001, 0.7, 4), (0.01, 1.0, 5), (0.5, 0.7, 2), (0.99, 0.7, 0)])
	def test_label_trainer(self, get_label_trainer_conf, mocker, iv, corr, remain):
		
		conf = get_label_trainer_conf
		conf["train_info"]["params"]["filter_params"]["common"]["threshold"] = iv
		conf["train_info"]["params"]["filter_params"]["correlation"]["correlation_threshold"] = corr

		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value="node-1"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		vfslt = VerticalFeatureSelectionLabelTrainer(conf)
		vfslt.node_id = "node-1"
		mocker.patch.object(
			vfslt.channels["feature_id_com"],
			"collect",
			return_value=[{
				"ZY8HUnfdEtpr6LIW": "sIcVutHjKxZyqUg1",
				"McqB4XwT3NSoWEyt": "FkYzRlp1ySwdnm0N"
			}]
		)
		vfslt.fit()
		with open("/opt/checkpoints/unit_test/feature_selection_guest.pkl", 'rb') as f:
			model = pickle.load(f)
			assert model["num_of_features"] == remain

	def test_rewrite_model(self, get_label_trainer_conf, mocker):
		conf = get_label_trainer_conf
		conf["input"]["model"] = {
			"type": "file",
			"path": "/opt/checkpoints/unit_test",
			"name": "binning_woe_iv_guest.json"
		}
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value="node-1"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		vfslt = VerticalFeatureSelectionLabelTrainer(conf)
		vfslt.node_id = "node-1"
		mocker.patch.object(
			vfslt.channels["feature_id_com"],
			"collect",
			return_value=[{
				"ZY8HUnfdEtpr6LIW": "sIcVutHjKxZyqUg1",
				"McqB4XwT3NSoWEyt": "FkYzRlp1ySwdnm0N"
			}]
		)
		vfslt.fit()
		with open("/opt/checkpoints/unit_test/feature_selection_guest.pkl", 'rb') as f:
			model = pickle.load(f)
			assert model["num_of_features"] == 4

		with open("/opt/checkpoints/unit_test/binning_woe_iv_guest.json", 'r') as f:
			model = json.load(f)
			assert len(model) == 1

	def test_trainer(self, get_trainer_conf, mocker):
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value="node-1"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		vfst = VerticalFeatureSelectionTrainer(get_trainer_conf)
		vfst.node_id = "node-2"

		def mock_feature_com():
			# 第一次通信在correlation_filter里，传递iv的id_mapping，构造反向映射
			# 后一次通信在结尾处，传递经过corr的id，留存结果
			if mock_feat.call_count <= 1:
				return ["ZY8HUnfdEtpr6LIW", "McqB4XwT3NSoWEyt"]
			else:
				return ["sIcVutHjKxZyqUg1", "FkYzRlp1ySwdnm0N"]

		mock_feat = mocker.patch.object(
			vfst.channels["feature_id_com"], "recv", side_effect=mock_feature_com
		)
		mocker.patch.object(
			vfst.channels["feature_id_com"], "send", return_value=0
		)
		vfst.fit()
		with open("/opt/checkpoints/unit_test/feature_selection_host.pkl", 'rb') as f:
			model = pickle.load(f)
			assert len(model) == 2
			assert 'x3' in model
			assert 'x4' in model
