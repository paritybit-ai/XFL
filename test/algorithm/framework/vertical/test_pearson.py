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


import json
import os
import shutil
import pickle

import numpy as np
import pandas as pd
import pytest

import service.fed_config
from algorithm.framework.vertical.pearson.label_trainer import VerticalPearsonLabelTrainer
from algorithm.framework.vertical.pearson.trainer import VerticalPearsonTrainer
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from algorithm.core.paillier_acceleration import embed
from common.communication.gRPC.python.commu import Commu

def prepare_data():
	label_list = [0] * 500 + [1] * 500
	np.random.shuffle(label_list)
	x0 = np.random.random(1000)
	df = pd.DataFrame({
		"y": label_list,
		"x0": x0,
		"x1": x0 + 0.01,
		"x2": np.random.random(1000),
		"x3": np.random.random(1000),
		"x4": np.random.random(1000)
	})
	df[["y", "x0", "x2"]].to_csv(
		"/opt/dataset/unit_test/train_guest.csv", index=True, index_label='id'
	)
	df[["x1", "x3", "x4"]].to_csv(
		"/opt/dataset/unit_test/train_host.csv", index=True, index_label='id'
	)


@pytest.fixture(scope="module", autouse=True)
def env():
	Commu.node_id="node-1"
	Commu.trainer_ids = ['node-1', 'node-2']
	Commu.scheduler_id = 'assist_trainer'
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
	with open("python/algorithm/config/vertical_pearson/label_trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train_guest.csv"
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
	yield conf


@pytest.fixture()
def get_trainer_conf():
	with open("python/algorithm/config/vertical_pearson/trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train_host.csv"
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
	yield conf


class TestVerticalPearsonTrainer:
	@pytest.mark.parametrize("node_id, encryption", [
		("node-2", "plain"), ("node-3", "plain"),
		("node-2", "paillier"), ("node-3", "paillier")
	])
	def test_trainer_3party(self, get_trainer_conf, mocker, node_id, encryption):
		sample_size = 200
		conf = get_trainer_conf
		conf["train_info"]["params"]["sample_size"] = sample_size
		if encryption == "plain":
			conf["train_info"]["params"]["encryption_params"] = {"plain": {}}

		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2", "node-3"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2", "node-3"]
		)

		def mock_context(use_pickle=False):
			encryption_params = get_trainer_conf.get("train_info").get("params").get(
				"encryption_params"
			)
			encryption = list(encryption_params.keys())[0]
			if encryption == "plain":
				return None
			elif encryption == "paillier":
				public_context = Paillier.context(
					encryption_params[encryption].get("key_bit_size"),
					encryption_params[encryption].get("djn_on")
				).to_public().serialize()
				return public_context

		mocker.patch.object(
			BroadcastChannel, "recv", side_effect=mock_context
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		vpt = VerticalPearsonTrainer(conf)
		vpt.node_id = node_id
		vpt.channels["trainer_com"]["node-2"] = DualChannel(
			name="trainer_com_node-2_node-3",
			ids=["node-3", "node-2"]
		)
		vpt.channels["trainer_com"]["node-3"] = DualChannel(
			name="trainer_com_node-3_node-2",
			ids=["node-2", "node-3"]
		)

		df = pd.read_csv("/opt/dataset/unit_test/train_guest.csv", index_col=0)

		def mock_result_feature():
			df['x0'] = (df['x0'] - df['x0'].mean()) / df['x0'].std()
			flag = True
			if vpt.encryption == "plain":
				other = df['x0'].to_numpy()
				return other, 'x0', flag
			elif vpt.encryption == "paillier":
				pc = Paillier.context_from(vpt.public_context.serialize())
				other = Paillier.encrypt(
					context=pc,
					data=df['x0'].to_numpy(),
					obfuscation=True,
					num_cores=1
				)
				return other, flag

		mocker.patch.object(
			vpt.channels["trainer_feature_com"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["trainer_com"]["node-2"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["trainer_com"]["node-3"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["sample_idx_com"], "recv", return_value=list(range(200))
		)
		vpt.fit()

		with open("/opt/checkpoints/unit_test/vertical_pearson_host.pkl", 'rb') as f:
			model = pickle.load(f)
			assert "corr" in model
			assert len(model["features"]) == 3
			assert model["feature_source"] == [node_id, node_id, node_id]
			assert "feature_mapping" in model

	def test_label_trainer(self, get_label_trainer_conf, mocker):
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		vpt = VerticalPearsonLabelTrainer(get_label_trainer_conf)
		vpt.node_id = "node-1"
		df1 = pd.read_csv("/opt/dataset/unit_test/train_guest.csv", index_col=0)
		del df1['y']
		df1 = (df1 - df1.mean()) / df1.std()
		df2 = pd.read_csv("/opt/dataset/unit_test/train_host.csv", index_col=0)
		df2 = (df2 - df2.mean()) / df2.std()

		def mock_result_feature():
			e = embed(df1.to_numpy().T)
			c = Paillier.encrypt(vpt.private_context, e)
			local_mat = np.array(df2.to_numpy() * 10 ** vpt.encryption_param.precision, dtype=int)
			corr = np.dot(local_mat.T, c).T
			return corr

		def mock_result_corr():
			corr = np.dot(df2.T, df2)
			corr /= len(df2)
			summary = {("node-2", "node-2"): corr}
			return summary

		mocker.patch.object(
			vpt.channels["trainer_feature_com"]["node-2"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["trainer_corr_com"]["node-2"], "recv", side_effect=mock_result_corr
		)
		vpt.fit()

		with open("/opt/checkpoints/unit_test/vertical_pearson_guest.pkl", 'rb') as f:
			model = pickle.load(f)
			assert "corr" in model
			assert len(model["features"]) == 5
			assert model["feature_source"] == ['node-1', 'node-1', 'node-2', 'node-2', 'node-2']
			np.testing.assert_almost_equal(model["corr"][0][2], np.dot(df1["x0"], df2["x1"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][0][3], np.dot(df1["x0"], df2["x3"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][0][4], np.dot(df1["x0"], df2["x4"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][2], np.dot(df1["x2"], df2["x1"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][3], np.dot(df1["x2"], df2["x3"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][4], np.dot(df1["x2"], df2["x4"]) / len(df1), decimal=3)

	def test_label_trainer_with_selection(self, get_label_trainer_conf, mocker):
		conf = get_label_trainer_conf
		conf["train_info"]["params"]["column_indexes"] = [2]
		conf["train_info"]["params"]["column_names"] = "x2"
		
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		vpt = VerticalPearsonLabelTrainer(conf)
		vpt.node_id = "node-1"
		df1 = pd.read_csv("/opt/dataset/unit_test/train_guest.csv", index_col=0)
		del df1['y']
		df1 = (df1 - df1.mean()) / df1.std()
		df2 = pd.read_csv("/opt/dataset/unit_test/train_host.csv", index_col=0)
		df2 = (df2 - df2.mean()) / df2.std()

		def mock_result_feature():
			e = embed(df1.to_numpy().T)
			c = Paillier.encrypt(vpt.private_context, e)
			local_mat = np.array(df2.to_numpy() * 10 ** vpt.encryption_param.precision, dtype=int)
			corr = np.dot(local_mat.T, c).T
			return corr

		def mock_result_corr():
			corr = np.dot(df2.T, df2)
			corr /= len(df2)
			summary = {("node-2", "node-2"): corr}
			return summary

		mocker.patch.object(
			vpt.channels["trainer_feature_com"]["node-2"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["trainer_corr_com"]["node-2"], "recv", side_effect=mock_result_corr
		)
		vpt.fit()

		with open("/opt/checkpoints/unit_test/vertical_pearson_guest.pkl", 'rb') as f:
			model = pickle.load(f)
			assert "corr" in model
			assert len(model["features"]) == 5
			assert model["feature_source"] == ['node-1', 'node-1', 'node-2', 'node-2', 'node-2']
			np.testing.assert_almost_equal(model["corr"][0][2], np.dot(df1["x0"], df2["x1"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][0][3], np.dot(df1["x0"], df2["x3"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][0][4], np.dot(df1["x0"], df2["x4"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][2], np.dot(df1["x2"], df2["x1"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][3], np.dot(df1["x2"], df2["x3"]) / len(df1), decimal=3)
			np.testing.assert_almost_equal(model["corr"][1][4], np.dot(df1["x2"], df2["x4"]) / len(df1), decimal=3)

	def test_label_trainer_fast(self, get_label_trainer_conf, mocker):
		sample_size = 200
		conf = get_label_trainer_conf
		conf["train_info"]["params"]["sample_size"] = sample_size
		conf["train_info"]["params"]["encryption_params"] = {"plain": {}}

		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		vpt = VerticalPearsonLabelTrainer(conf)
		vpt.node_id = "node-1"
		df1 = pd.read_csv("/opt/dataset/unit_test/train_guest.csv", index_col=0)
		del df1['y']
		df1 = (df1 - df1.mean()) / df1.std()
		df2 = pd.read_csv("/opt/dataset/unit_test/train_host.csv", index_col=0)
		df2 = (df2 - df2.mean()) / df2.std()

		def mock_result_feature():
			if mock_result.call_count > 1:
				f = 'x2'
			else:
				f = 'x0'
			local_mat = df2.to_numpy()
			corr = np.dot(local_mat.T, df1[f])
			return corr

		def mock_result_corr():
			corr = np.dot(df2.T, df2)
			corr /= len(df2)
			summary = {("node-2", "node-2"): corr}
			return summary

		mock_result = mocker.patch.object(
			vpt.channels["trainer_feature_com"]["node-2"], "recv", side_effect=mock_result_feature
		)
		mocker.patch.object(
			vpt.channels["trainer_corr_com"]["node-2"], "recv", side_effect=mock_result_corr
		)
		vpt.fit()

		with open("/opt/checkpoints/unit_test/vertical_pearson_guest.pkl", 'rb') as f:
			model = pickle.load(f)
			assert "corr" in model
			assert len(model["features"]) == 5
			assert model["feature_source"] == ['node-1', 'node-1', 'node-2', 'node-2', 'node-2']
			np.testing.assert_almost_equal(model["corr"][0][2], np.dot(df1["x0"], df2["x1"]) / sample_size, decimal=2)
			np.testing.assert_almost_equal(model["corr"][0][3], np.dot(df1["x0"], df2["x3"]) / sample_size, decimal=2)
			np.testing.assert_almost_equal(model["corr"][0][4], np.dot(df1["x0"], df2["x4"]) / sample_size, decimal=2)
			np.testing.assert_almost_equal(model["corr"][1][2], np.dot(df1["x2"], df2["x1"]) / sample_size, decimal=2)
			np.testing.assert_almost_equal(model["corr"][1][3], np.dot(df1["x2"], df2["x3"]) / sample_size, decimal=2)
			np.testing.assert_almost_equal(model["corr"][1][4], np.dot(df1["x2"], df2["x4"]) / sample_size, decimal=2)

	def test_trainer(self, get_trainer_conf, mocker):
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["node-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["node-2"]
		)

		def mock_context(use_pickle=False):
			encryption_params = get_trainer_conf.get("train_info").get("params").get(
				"encryption_params"
			)
			encryption = list(encryption_params.keys())[0]
			if encryption == "plain":
				return None
			elif encryption == "paillier":
				public_context = Paillier.context(
					encryption_params[encryption].get("key_bit_size"),
					encryption_params[encryption].get("djn_on")
				).to_public().serialize()
				return public_context

		mocker.patch.object(
			BroadcastChannel, "recv", side_effect=mock_context
		)
		mocker.patch.object(
			BroadcastChannel, "broadcast", return_value=0
		)
		vpt = VerticalPearsonTrainer(get_trainer_conf)
		vpt.node_id = "node-2"

		df = pd.read_csv("/opt/dataset/unit_test/train_guest.csv", index_col=0)

		def mock_result_feature():
			df['x0'] = (df['x0'] - df['x0'].mean()) / df['x0'].std()
			flag = True
			if vpt.encryption == "plain":
				other = df['x0'].to_numpy()
				return other, 'x0', flag
			elif vpt.encryption == "paillier":
				pc = Paillier.context_from(vpt.public_context.serialize())
				other = Paillier.encrypt(
					context=pc,
					data=df['x0'].to_numpy(),
					obfuscation=True,
					num_cores=1
				)
				return other, flag

		mocker.patch.object(
			vpt.channels["trainer_feature_com"], "recv", side_effect=mock_result_feature
		)
		vpt.fit()

		with open("/opt/checkpoints/unit_test/vertical_pearson_host.pkl", 'rb') as f:
			model = pickle.load(f)
			assert "corr" in model
			assert len(model["features"]) == 3
			assert model["feature_source"] == ['node-2', 'node-2', 'node-2']
			assert "feature_mapping" in model
