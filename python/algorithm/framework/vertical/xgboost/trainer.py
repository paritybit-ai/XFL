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
from typing import Any

import numpy as np
import ray

from algorithm.core.encryption_param import PaillierParam, PlainParam
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.crypto.paillier.utils import get_core_num
from common.utils.logger import logger
from common.utils.tree_transfer import trainer_tree_transfer
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from .base import VerticalXgboostBase
from .decision_tree_trainer import VerticalDecisionTreeTrainer


class VerticalXgboostTrainer(VerticalXgboostBase):
	def __init__(self, train_conf: dict, *args, **kwargs):
		super().__init__(train_conf, is_label_trainer=False, *args, **kwargs)

		self.channels = {}
		self.channels["encryption_context"] = BroadcastChannel(name="encryption_context")
		self.channels["individual_grad_hess"] = BroadcastChannel(name="individual_grad_hess")
		self.channels["tree_node"] = BroadcastChannel(name="tree_node")

		self.channels["summed_grad_hess"] = DualChannel(name="summed_grad_hess_" + FedConfig.node_id,
														ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
		self.channels["min_split_info"] = DualChannel(name="min_split_info_" + FedConfig.node_id,
													  ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
		self.channels["sample_index_after_split"] = DualChannel(name="sample_index_after_split_" + FedConfig.node_id,
																ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
		self.channels["val_com"] = DualChannel(name="val_com_" + FedConfig.node_id,
											   ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
		self.channels["early_stop_com"] = DualChannel(name="early_stop_com_" + FedConfig.node_id,
													  ids=FedConfig.get_label_trainer() + [FedConfig.node_id])

		if isinstance(self.xgb_config.encryption_param, (PlainParam, type(None))):
			self.public_context = None
		elif isinstance(self.xgb_config.encryption_param, PaillierParam):
			self.public_context = self.channels["encryption_context"].recv(use_pickle=False)
			self.public_context = Paillier.context_from(self.public_context)
		else:
			raise TypeError(f"Encryption param type {type(self.xgb_config.encryption_param)} not valid.")
		self.export_conf = [{
			"class_name": "VerticalXGBooster",
			"identity": self.identity,
			"filename": self.output.get("model", {"name": "vertical_xgboost_host.pt"})["name"]
		}]

		ray.init(num_cpus=get_core_num(self.xgb_config.max_num_cores),
				 ignore_reinit_error=True)

	def fit(self):
		nodes_list = {}
		for tree_idx in range(self.xgb_config.num_trees):
			# training section
			logger.info("Tree {} start training.".format(tree_idx))
			sampled_features, feature_id_mapping = self.col_sample()
			trainer = VerticalDecisionTreeTrainer(tree_param=self.xgb_config,
												  features=sampled_features,
												  split_points=self.split_points,
												  channels=self.channels,
												  encryption_context=self.public_context,
												  feature_id_mapping=feature_id_mapping,
												  tree_index=tree_idx)
			nodes = trainer.fit()
			logger.info("Tree {} training done.".format(tree_idx))
			if self.channels["early_stop_com"].recv():
				logger.info("trainer early stopped.")
				break

			nodes_list.update(trainer_tree_transfer(nodes))
			if self.xgb_config.run_goss:
				for _, (x, _) in enumerate(self.train_dataset):
					node_feedback = {}
					for node_idx, node in nodes.items():
						node_feedback[node_idx] = x[:, node.split_info.feature_idx] < node.split_info.split_point
					self.channels["val_com"].send(node_feedback)

			# valid section
			logger.info("trainer: Validation on tree {} start.".format(tree_idx))
			for _, (x, _) in enumerate(self.val_dataset):
				node_feedback = {}
				for node_idx, node in nodes.items():
					node_feedback[node_idx] = x[:, node.split_info.feature_idx] < node.split_info.split_point
				self.channels["val_com"].send(node_feedback)
			if self.channels["early_stop_com"].recv():
				logger.info("trainer early stopped.")
				break
			logger.info("Validation on tree {} done.".format(tree_idx))

			if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get(
					"save_frequency") == 0:
				self.save(nodes_list, epoch=tree_idx + 1)
		# model preserve
		self.save(nodes_list, final=True)
		ray.shutdown()

	def col_sample(self) -> tuple[Any, dict]:
		col_size = self.train_features.shape[1]
		if 0 < self.xgb_config.subsample_feature_rate <= 1:
			sample_num = int(col_size * self.xgb_config.subsample_feature_rate)
		else:
			sample_num = col_size
		sampled_idx = np.sort(np.random.choice(col_size, sample_num, replace=False))
		feature_id_mapping = {a: b for a, b in enumerate(sampled_idx)}
		sampled_features = self.train_features.iloc[:, sampled_idx]
		return sampled_features, feature_id_mapping

	def save(self, node_list, epoch: int = None, final: bool = False):
		if final:
			save_model_config(stage_model_config=self.export_conf, save_path=Path(self.output.get("model")["path"]))

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

		xgb_output = {
			# "nodes": node_list,
			"nodes": {k: v for k, v in sorted(node_list.items())},
			"num_trees": self.xgb_config.num_trees
		}

		with open(model_path, 'wb') as f:
			pickle.dump(xgb_output, f)
		logger.info("model saved as: {}.".format(model_path))

	def load_model(self):
		model_path = Path(
			self.input.get("pretrain_model", {}).get("path", ''),
			self.input.get("pretrain_model", {}).get("name", '')
		)
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
		self.xgb_config.num_trees = model["num_trees"]
		nodes = model["nodes"]
		return nodes

	def predict(self):
		nodes = self.load_model()
		for i in range(self.xgb_config.num_trees):
			for _, (x, _) in enumerate(self.test_dataset):
				node_feedback = {}
				for node_idx, node in nodes.items():
					node_feedback[node_idx] = x[:, node.feature_idx] < node.split_point
				self.channels["val_com"].send(node_feedback)
