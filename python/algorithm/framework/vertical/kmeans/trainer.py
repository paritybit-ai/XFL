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
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.checker.matcher import get_matched_config
from common.checker.x_types import All
from common.communication.gRPC.python.channel import DualChannel
from common.utils.logger import logger
from common.utils.utils import update_dict
from service.fed_config import FedConfig
from service.fed_node import FedNode
from .api import get_table_agg_trainer_inst
from .base import VerticalKmeansBase


class VerticalKmeansTrainer(VerticalKmeansBase):
	def __init__(self, train_conf: dict, *args, **kwargs):
		"""

		Args:
			train_conf:
			*args:
			**kwargs:
		"""
		self.channels = {}
		self.channels["sync"] = DualChannel(
			name="sync_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		conf = self._sync_config(train_conf)
		update_dict(train_conf, conf)
		super().__init__(train_conf, label=False, *args, **kwargs)

		self.dist_table = None
		self.cluster_result = None
		self.local_tol = 0.0

		self.channels["cluster_result"] = DualChannel(
			name="cluster_res_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.channels["tolerance"] = DualChannel(
			name="tolerance_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.channels["converged_flag"] = DualChannel(
			name="converged_flag_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.channels["check_data_com"] = DualChannel(
			name="check_data_com_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.table_agg_executor = get_table_agg_trainer_inst(
			sec_conf=self.encryption, trainer_ids=FedConfig.get_label_trainer() +
													FedConfig.get_trainer()
		)

	def _sync_config(self, config):
		sync_rule = {
			"train_info": All()
		}

		config_to_sync = get_matched_config(config, sync_rule)
		self.channels["sync"].send(config_to_sync)
		conf = self.channels["sync"].recv()
		return conf

	def init_centers(self):
		"""
		Initialize cluster centers
		Returns:

		"""
		self.channels["init_center"] = DualChannel(
			name="init_center_" + FedNode.node_id, ids=[FedNode.node_id] + FedConfig.get_label_trainer()
		)

		if self.init == "random":
			center_ids = self.channels["init_center"].recv()
			return center_ids
		elif self.init == "kmeans++":
			center_ids = []
			while len(center_ids) < self.k:
				if len(center_ids) >= 1:
					dist_table = self.distance_table(self.train_features.iloc[center_ids])
					self.table_agg_executor.send(dist_table)
				center_ids = self.channels["init_center"].recv()
			return center_ids

	def check_data(self):
		m, n = len(self.train_ids), len(self.train_features.columns)
		self.channels["check_data_com"].send((m, n))

	def fit(self):
		logger.info("vertical K-means trainer start.")
		self.check_data()
		np.random.seed(self.random_seed)

		center_ids = self.init_centers()
		logger.info("{}::initialized centers.".format(self.identity))
		self.cluster_centers = self.train_features.iloc[center_ids]

		iter_ = 0
		for iter_ in range(self.max_iter):

			if iter_ <= 0:
				self.dist_table = self.distance_table(self.cluster_centers)
				self.table_agg_executor.send(self.dist_table)

			self.cluster_result = self.channels["cluster_result"].recv()
			centers = self.calc_centers(
				self.cluster_centers, self.cluster_result)

			self.local_tol = self.calc_tolerance(self.cluster_centers, centers)
			self.channels["tolerance"].send(self.local_tol)
			self.is_converged = self.channels["converged_flag"].recv()

			self.cluster_centers = centers
			self.dist_table = self.distance_table(self.cluster_centers)
			self.table_agg_executor.send(self.dist_table)

			center_dist = self.distance_between_centers(self.cluster_centers)
			self.table_agg_executor.send(center_dist)

			if self.is_converged:
				break

		self.save(epoch=iter_, final=True)

	def save(self, epoch: int = None, final: bool = False):
		"""

		Args:
			epoch:
			final:

		Returns:

		"""
		save_dir = str(Path(self.output.get("path")))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		model_name = self.output.get("model", {}).get("name", "")
		model_path = Path(save_dir, model_name)

		kmeans_output = {
			"k": self.k,
			"iter": epoch,
			"is_converged": self.is_converged,
			"tol": self.tol,
			"cluster_centers": self.cluster_centers
		}
		with open(model_path, 'wb') as f:
			pickle.dump(kmeans_output, f)
		logger.info("model saved as: {}.".format(model_path))

		result_dataframe = pd.DataFrame(
			{
				"id": self.train_ids.to_numpy(),
				"cluster_result": self.cluster_result
			}
		)
		result_name = self.output.get("result", {}).get("name", "")
		result_path = Path(save_dir, result_name)
		result_dataframe.to_csv(result_path, header=True, index=False)
		logger.info("result saved as: {}.".format(result_path))

		summary_df = result_dataframe.groupby(
			"cluster_result").size().to_frame("count")
		summary_df = summary_df.reset_index()
		summary_name = self.output.get("summary", {}).get("name", "")
		summary_path = Path(save_dir, summary_name)
		summary_df.to_csv(summary_path, header=True, index=False)
		logger.info("summary info saved to: {}.".format(summary_path))
