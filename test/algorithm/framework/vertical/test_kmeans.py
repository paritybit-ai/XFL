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


import collections
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import davies_bouldin_score

import service.fed_config
from algorithm.framework.vertical.kmeans.assist_trainer import \
	VerticalKmeansAssistTrainer
from algorithm.framework.vertical.kmeans.table_agg_base import (
	TableAggregatorAbstractAssistTrainer, TableAggregatorAbstractTrainer)
from algorithm.framework.vertical.kmeans.table_agg_otp import (
	TableAggregatorOTPAssistTrainer, TableAggregatorOTPTrainer)
from algorithm.framework.vertical.kmeans.table_agg_plain import (
	TableAggregatorPlainAssistTrainer, TableAggregatorPlainTrainer)
from algorithm.framework.vertical.kmeans.trainer import VerticalKmeansTrainer
from common.communication.gRPC.python.channel import (BroadcastChannel, DualChannel)
from common.communication.gRPC.python.commu import Commu

def prepare_data():
	label_list = [0, 1, 2, 3, 4] * 200
	np.random.shuffle(label_list)
	case_df = pd.DataFrame({
		"y": label_list,
		"x0": np.random.random(1000) * 0.2 + np.array(label_list),
		"x1": np.random.random(1000)
	})
	case_df[['y', 'x0', 'x1']].to_csv(
		"/opt/dataset/unit_test/train_guest.csv", index=True, index_label='id'
	)
	case_df[['x0', 'x1']].to_csv(
		"/opt/dataset/unit_test/train_host.csv", index=True, index_label='id'
	)
	case_df[['x0', 'x1']].to_csv(
		"/opt/dataset/unit_test/train_guest_without_id.csv", index=False
	)
	case_df[['x0', 'x1']].to_csv(
		"/opt/dataset/unit_test/train_host_without_id.csv", index=False
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
	with open("python/algorithm/config/vertical_kmeans/label_trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train_guest.csv"
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
	yield conf


@pytest.fixture()
def get_trainer_conf():
	with open("python/algorithm/config/vertical_kmeans/trainer.json") as f:
		conf = json.load(f)
		conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
		conf["input"]["trainset"][0]["name"] = "train_host.csv"
		conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
	yield conf


@pytest.fixture()
def get_scheduler_conf():
	with open("python/algorithm/config/vertical_kmeans/assist_trainer.json") as f:
		conf = json.load(f)
	yield conf


class TestVerticalKmeansTrainer:
	@pytest.mark.parametrize("computing_engine", ["local", "spark"])
	def test_label_trainer(self, get_label_trainer_conf, computing_engine, mocker):
		conf = get_label_trainer_conf
		conf["computing_engine"] = computing_engine
		# mock 类初始化需要的函数，避免建立通信通道时报错
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "send"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["trainer-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["trainer-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_assist_trainer", return_value="scheduler"
		)

		# 定义初始化类
		vkt = VerticalKmeansTrainer(conf)

		# mock scheduler侧的接口返回
		def mock_get_cluster():
			return VerticalKmeansAssistTrainer.get_cluster(vkt.dist_table)

		def mock_converged_flag():
			return vkt.local_tol < vkt.tol

		mocker.patch.object(
			vkt.channels["cluster_result"], "recv", side_effect=mock_get_cluster
		)
		mocker.patch.object(
			vkt.channels["converged_flag"], "recv", side_effect=mock_converged_flag
		)
		vkt.fit()

		# 是否能正常收敛
		assert vkt.is_converged

		assert os.path.exists("/opt/checkpoints/unit_test/vertical_kmeans_guest.pkl")
		with open("/opt/checkpoints/unit_test/vertical_kmeans_guest.pkl", "rb") as f:
			model = pickle.load(f)
			assert model["k"] == vkt.k
			assert model["iter"] <= vkt.max_iter
			assert model["is_converged"]
			assert model["tol"] == vkt.tol
			assert len(model["cluster_centers"]) == vkt.k

	def test_label_trainer_only_features(self, get_label_trainer_conf, mocker):
		conf = get_label_trainer_conf
		conf["input"]["trainset"][0]["has_id"] = False
		conf["input"]["trainset"][0]["has_label"] = False
		# mock 类初始化需要的函数，避免建立通信通道时报错
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "send"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["trainer-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["trainer-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_assist_trainer", return_value="scheduler"
		)

		# 定义初始化类
		vkt = VerticalKmeansTrainer(conf)

		# mock scheduler侧的接口返回
		def mock_get_cluster():
			return VerticalKmeansAssistTrainer.get_cluster(vkt.dist_table)

		def mock_converged_flag():
			return vkt.local_tol < vkt.tol

		mocker.patch.object(
			vkt.channels["cluster_result"], "recv", side_effect=mock_get_cluster
		)
		mocker.patch.object(
			vkt.channels["converged_flag"], "recv", side_effect=mock_converged_flag
		)
		vkt.fit()

		# 是否能正常收敛
		assert vkt.is_converged

		assert os.path.exists("/opt/checkpoints/unit_test/vertical_kmeans_guest.pkl")
		with open("/opt/checkpoints/unit_test/vertical_kmeans_guest.pkl", "rb") as f:
			model = pickle.load(f)
			assert model["k"] == vkt.k
			assert model["iter"] <= vkt.max_iter
			assert model["is_converged"]
			assert model["tol"] == vkt.tol
			assert len(model["cluster_centers"]) == vkt.k

	@pytest.mark.parametrize("computing_engine", ["local", "spark"])
	def test_trainer(self, get_trainer_conf, computing_engine, mocker):
		conf = get_trainer_conf
		conf["computing_engine"] = computing_engine
		# mock 类初始化需要的函数，避免建立通信通道时报错
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "send"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["trainer-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["trainer-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_assist_trainer", return_value="scheduler"
		)

		# 定义初始化类
		vkt = VerticalKmeansTrainer(get_trainer_conf)

		# 初始化类中心
		init_centers = np.random.choice(1000, vkt.k, replace=False)
		mocker.patch.object(
			vkt.channels["init_center"], "recv", return_value=init_centers
		)

		# mock scheduler侧的接口返回
		def mock_get_cluster():
			return VerticalKmeansAssistTrainer.get_cluster(vkt.dist_table)

		def mock_converged_flag():
			return vkt.local_tol < vkt.tol

		mocker.patch.object(
			vkt.channels["cluster_result"], "recv", side_effect=mock_get_cluster
		)
		mocker.patch.object(
			vkt.channels["converged_flag"], "recv", side_effect=mock_converged_flag
		)
		vkt.fit()

		# 是否能正常收敛
		assert vkt.is_converged

		assert os.path.exists("/opt/checkpoints/unit_test/vertical_kmeans_host.pkl")
		with open("/opt/checkpoints/unit_test/vertical_kmeans_host.pkl", "rb") as f:
			model = pickle.load(f)
			assert model["k"] == vkt.k
			assert model["iter"] <= vkt.max_iter
			assert model["is_converged"]
			assert model["tol"] == vkt.tol
			assert len(model["cluster_centers"]) == vkt.k

		# 检查输出
		assert os.path.exists("/opt/checkpoints/unit_test/cluster_result.csv")
		if computing_engine == "local":
			df = pd.read_csv("/opt/checkpoints/unit_test/cluster_result.csv")
			assert (df["id"] == vkt.train_ids).all()

	# assert (df["cluster_label"] == vkt.cluster_result).all()

	@pytest.mark.parametrize("computing_engine", ["local", "spark"])
	def test_scheduler(self, get_scheduler_conf, computing_engine, mocker):
		conf = get_scheduler_conf
		conf["computing_engine"] = computing_engine
		# mock 类初始化需要的函数，避免建立通信通道时报错
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			TableAggregatorOTPAssistTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["trainer-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["trainer-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_assist_trainer", return_value="scheduler"
		)

		# 定义初始化类
		vks = VerticalKmeansAssistTrainer(conf)

		# mock trainer的发送结果
		# tolerance
		mocker.patch.object(
			DualChannel, "recv", return_value=1.0
		)
		# distance table
		dist_table = torch.tensor(np.random.random((1000, vks.k)))
		# center dist
		center_dist = torch.tensor(np.random.random(vks.k * (vks.k - 1)))

		def mock_aggregate():
			if mock_agg.call_count > 1 and mock_agg.call_count % 2 == 1:
				return center_dist
			else:
				return dist_table

		mock_agg = mocker.patch.object(
			vks.dist_table_agg_executor, "aggregate", side_effect=mock_aggregate
		)
		vks.fit()

	def test_calc_dbi(self, get_scheduler_conf, get_label_trainer_conf, mocker):
		# mock 类初始化需要的函数，避免建立通信通道时报错
		mocker.patch.object(
			DualChannel, "__init__", return_value=None
		)
		mocker.patch.object(
			DualChannel, "send", return_value=0
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			TableAggregatorOTPAssistTrainer, "__init__", return_value=None
		)
		mocker.patch.object(
			TableAggregatorOTPTrainer, "send"
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_label_trainer", return_value=["trainer-1"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_trainer", return_value=["trainer-2"]
		)
		mocker.patch.object(
			service.fed_config.FedConfig, "get_assist_trainer", return_value="scheduler"
		)

		vkt = VerticalKmeansTrainer(get_label_trainer_conf)
		vks = VerticalKmeansAssistTrainer(get_scheduler_conf)

		# 检查指标
		center_ids = vkt.init_centers()
		cluster_centers = vkt.train_features.iloc[center_ids]
		dist_table = vkt.distance_table(cluster_centers)
		cluster_result = vks.get_cluster(dist_table)

		centers = vkt.calc_centers(cluster_centers, cluster_result)
		center_dist = vkt.distance_between_centers(centers)

		mocker.patch.object(
			vks.dist_table_agg_executor, "aggregate", return_value=center_dist
		)
		vks.cluster_count_list = vks.calc_cluster_count(cluster_result)
		dist_table = vkt.distance_table(centers)

		vks.calc_dbi(dist_table, cluster_result, 0)

		dbi_score = davies_bouldin_score(vkt.train_features.to_numpy(), cluster_result)
		assert np.abs(vks.DBI - dbi_score) < 1e-5

		# 验证当一族结果为空时，DBI的计算
		cluster_result_missing = []
		for _ in cluster_result:
			if _ != 1:
				cluster_result_missing.append(_)
			else:
				cluster_result_missing.append(0)
		# 重新计算簇中心坐标
		centers = vkt.calc_centers(cluster_centers, cluster_result_missing)
		center_dist = vkt.distance_between_centers(centers)
		mocker.patch.object(
			vks.dist_table_agg_executor, "aggregate", return_value=center_dist
		)
		vks.cluster_count_list = vks.calc_cluster_count(cluster_result_missing)
		dist_table = vkt.distance_table(centers)

		vks.calc_dbi(dist_table, cluster_result_missing, 1)

		dbi_score = davies_bouldin_score(vkt.train_features.to_numpy(), cluster_result_missing)
		assert np.abs(vks.DBI - dbi_score) < 1e-5

	def test_table_agg_base(self):
		table_trainer = TableAggregatorAbstractTrainer()
		table_trainer.send(pd.DataFrame({"x": [1, 2, 3]}))

		table_scheduler = TableAggregatorAbstractAssistTrainer()
		table_scheduler.aggregate()

	def test_table_agg_plain(self, mocker):
		pd_table = pd.DataFrame({"x": [1, 2, 3]})
		mocker.patch.object(
			BroadcastChannel, "send", return_value=0
		)
		mocker.patch.object(
			BroadcastChannel, "collect", return_value=[pd_table, pd_table]
		)

		table_trainer = TableAggregatorPlainTrainer(sec_conf={"method": "plain"})
		table_trainer.send(pd_table)

		table_scheduler = TableAggregatorPlainAssistTrainer(sec_conf={"method": "plain"})
		aggregated_table = table_scheduler.aggregate()
		assert aggregated_table["x"].iloc[2] == 6

	def test_table_agg_otp(self, mocker):
		pd_table = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
		table = torch.tensor(pd_table.values)
		mocker.patch.object(
			BroadcastChannel, "send", return_value=0
		)
		mocker.patch.object(
			BroadcastChannel, "collect", return_value=[pd_table, pd_table]
		)

		encryption = {
			"method": "otp",
			"key_bitlength": 128,
			"data_type": "torch.Tensor",
			"key_exchange": {
				"key_bitlength": 3072,
				"optimized": True
			},
			"csprng": {
				"name": "hmac_drbg",
				"method": "sha512"
			}
		}

		table_trainer = TableAggregatorOTPTrainer(sec_conf=encryption, trainer_ids=[])
		# table_trainer.send(table)

		table_scheduler = TableAggregatorOTPAssistTrainer(sec_conf=encryption, trainer_ids=[])
		# aggregated_table = table_scheduler.aggregate()
		# assert aggregated_table["x"].iloc[2] == 6