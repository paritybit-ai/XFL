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
from typing import Dict

import numpy as np
import pandas as pd

from common.communication.gRPC.python.channel import DualChannel
from common.utils.logger import logger
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
		super().__init__(train_conf, label=False, *args, **kwargs)
		self.channels = {}
		self.dist_table = None
		self.cluster_result = None
		self.local_tol = 0.0

		if self.identity == "label_trainer":
			# current node is a label trainer
			init_center_chan: Dict[str, DualChannel] = {}
			for party_id in FedConfig.get_trainer():
				init_center_chan[party_id] = DualChannel(
					name="init_center_" + party_id, ids=[FedNode.node_id, party_id]
				)
			self.channels["init_center"] = init_center_chan
		elif self.identity == "trainer":
			# current node is a trainer
			self.channels["init_center"] = DualChannel(
				name="init_center_" + FedNode.node_id, ids=[FedNode.node_id] + FedConfig.get_label_trainer()
			)

		self.channels["cluster_result"] = DualChannel(
			name="cluster_res_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.channels["tolerance"] = DualChannel(
			name="tolerance_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.channels["converged_flag"] = DualChannel(
			name="converged_flag_" + FedNode.node_id, ids=[FedConfig.get_assist_trainer(), FedNode.node_id]
		)
		self.table_agg_executor = get_table_agg_trainer_inst(
			sec_conf=self.encryption, trainer_ids=FedConfig.get_label_trainer() + FedConfig.get_trainer()
		)

	def init_centers(self):
		"""
		Initialize cluster centers
		Returns:

		"""
		random_list = list(np.random.choice(len(self.train_features), self.k, replace=False))
		return random_list

	def fit(self):
		logger.info("vertical K-means trainer start.")
		np.random.seed(self.random_seed)

		center_ids = []
		if self.identity == "label_trainer":
			center_ids = self.init_centers()
			for party_id in FedConfig.get_trainer():
				self.channels["init_center"][party_id].send(center_ids)
		elif self.identity == "trainer":
			center_ids = self.channels["init_center"].recv()
		self.cluster_centers = self.train_features.iloc[center_ids]

		iter_ = 0
		for iter_ in range(self.max_iter):

			if iter_ <= 0:
				self.dist_table = self.distance_table(self.cluster_centers)
				self.table_agg_executor.send(self.dist_table)

			self.cluster_result = self.channels["cluster_result"].recv()
			centers = self.calc_centers(self.cluster_centers, self.cluster_result)

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
		save_dir = str(Path(self.output.get("model")["path"]))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		model_name_list = self.output.get("model")["name"].split(".")
		name_prefix, name_postfix = ".".join(model_name_list[:-1]), model_name_list[-1]
		if not final and epoch:
			model_name = name_prefix + "_{}".format(epoch) + "." + name_postfix
		else:
			model_name = name_prefix + "." + name_postfix
		model_path = os.path.join(save_dir, model_name)

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

		rdf = pd.DataFrame(
			{
				"id": self.train_ids.to_numpy(),
				"cluster_result": self.cluster_result
			}
		)
		if final:
			file_name = os.path.join(save_dir, "cluster_result.csv")
		else:
			file_name = os.path.join(save_dir, "cluster_result.epoch_{}".format(epoch))
		rdf.to_csv(file_name, header=True, index=False)
