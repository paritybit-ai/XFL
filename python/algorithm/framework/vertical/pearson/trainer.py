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
from pathlib import Path
import pickle

import numpy as np

from common.utils.utils import update_dict
from service.fed_config import FedConfig
from service.fed_node import FedNode
from .base import VerticalPearsonBase
from algorithm.core.encryption_param import PlainParam, PaillierParam
from common.crypto.paillier.paillier import Paillier
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.utils.logger import logger
from algorithm.core.paillier_acceleration import embed


class VerticalPearsonTrainer(VerticalPearsonBase):
	def __init__(self, train_conf: dict):
		"""

		Args:
			train_conf:
		"""
		self.sync_channel = BroadcastChannel(name="sync")
		conf = self._sync_config()
		update_dict(train_conf, conf)
		super().__init__(train_conf, label=False)
		self.channels = {}
		self.node_id = FedNode.node_id
		self.channels = dict()
		self.channels["encryption_context"] = BroadcastChannel(name="encryption_context")
		self.channels["sample_idx_com"] = BroadcastChannel(name="sample_idx_com")
		self.channels["trainer_feature_com"] = DualChannel(
			name="trainer_feature_com_" + self.node_id,
			ids=[self.node_id] + FedConfig.get_label_trainer()
		)
		self.channels["trainer_corr_com"] = DualChannel(
			name="trainer_corr_com_" + self.node_id,
			ids=[self.node_id] + FedConfig.get_label_trainer()
		)
		self.channels["trainer_com"] = dict()
		self.trainers = FedConfig.get_trainer()
		for trainer_id in self.trainers:
			if trainer_id != self.node_id:
				if self.node_id not in self.trainers or trainer_id not in self.trainers:
					continue
				elif self.trainers.index(trainer_id) < self.trainers.index(self.node_id):
					self.channels["trainer_com"][trainer_id] = DualChannel(
						name="trainer_com_" + trainer_id + "_" + self.node_id,
						ids=[self.node_id, trainer_id]
					)
				else:
					self.channels["trainer_com"][trainer_id] = DualChannel(
						name="trainer_com_" + self.node_id + "_" + trainer_id,
						ids=[self.node_id, trainer_id]
					)

		if isinstance(self.encryption_param, (PlainParam, type(None))):
			self.public_context = None
		elif isinstance(self.encryption_param, PaillierParam):
			self.public_context = self.channels["encryption_context"].recv(use_pickle=False)
			self.public_context = Paillier.context_from(self.public_context)
		else:
			raise TypeError(f"Encryption param type {type(self.encryption_param)} not valid.")
		self.feature_mapping = dict()

	def _sync_config(self):
		config = self.sync_channel.recv()
		return config

	def fit(self):
		logger.info("vertical pearson trainer start.")

		data = self._select_columns()
		if self.sample_size is not None and self.sample_size < len(data):
			logger.info("sampled %d data." % self.sample_size)
			sample_ids = self.channels["sample_idx_com"].recv()
			data = data.iloc[sample_ids]

		data = data.apply(self.standardize)
		n = len(data)
		feature_names = data.columns.to_list()
		self.feature_mapping = self.string_encryption(feature_names)

		local_corr = np.dot(data.T, data)
		local_corr /= n
		self._local_summary["corr"] = local_corr
		self._local_summary["features"] = feature_names
		self._local_summary["num_features"] = {
			self.node_id: len(feature_names)
		}
		self._local_summary["feature_source"] = [self.node_id] * len(feature_names)
		self._local_summary["feature_mapping"] = self.feature_mapping
		self._summary[(self.node_id, self.node_id)] = local_corr

		feature_flag = self.channels["trainer_feature_com"].recv()
		# remote_corr = pd.DataFrame()
		j = 0
		local_mat = np.array([])
		if isinstance(self.encryption_param, PlainParam):
			local_mat = data.to_numpy()
		elif isinstance(self.encryption_param, PaillierParam):

			local_mat = np.array(data.to_numpy() * 10 ** self.encryption_param.precision, dtype=int)
		while not feature_flag:
			other, feature_flag = self.channels["trainer_feature_com"].recv()
			remote_corr = np.dot(local_mat.T, other)

			self.channels["trainer_feature_com"].send(remote_corr)
			logger.info("trainer calculated {} feature from label_trainer.".format(j + 1))
			j += 1

		self.channels["trainer_feature_com"].send([self.feature_mapping[f] for f in feature_names])

		for i in range(self.trainers.index(self.node_id)):
			trainer_id = self.trainers[i]
			flag = self.channels["trainer_com"][trainer_id].recv()
			j = 0
			corr_mat = []
			pack_nums = []
			while not flag:
				other, pack_num, flag = self.channels["trainer_com"][trainer_id].recv()
				remote_corr = np.dot(local_mat.T, other)
				self.channels["trainer_com"][trainer_id].send(True)
				corr_mat.append(remote_corr)
				pack_nums.append(pack_num)
				logger.info("trainer {} calculated {} feature from trainer {}.".format(self.node_id, j + 1, trainer_id))
				j += 1
			self._summary[(self.node_id, trainer_id)] = (np.array(corr_mat), pack_nums)

		for j in range(self.trainers.index(self.node_id) + 1, len(self.trainers)):
			trainer_id = self.trainers[j]
			if len(feature_names):
				self.channels["trainer_com"][trainer_id].send(False)
			else:
				self.channels["trainer_com"][trainer_id].send(True)
			if isinstance(self.encryption_param, (PlainParam, type(None))):
				cnt = 0
				for f in feature_names:
					encrypted_data = data[f].to_numpy()
					if f != feature_names[-1]:
						self.channels["trainer_com"][trainer_id].send((encrypted_data, 1, False))
					else:
						self.channels["trainer_com"][trainer_id].send((encrypted_data, 1, True))
					cnt += 1
					logger.info(
						"trainer {} encrypted {} features to trainer {}.".format(
							self.node_id, cnt, trainer_id))
					self.channels["trainer_com"][trainer_id].recv()
			elif isinstance(self.encryption_param, PaillierParam):
				for i in range(0, len(feature_names), self.num_embed):
					embedded_data = embed(data.iloc[:, i:(i + self.num_embed)].to_numpy().T)
					encrypted_data = Paillier.encrypt(
						context=self.public_context,
						data=embedded_data,
						obfuscation=True,
						num_cores=self.max_num_cores
					)
					if i + self.num_embed >= len(feature_names):
						self.channels["trainer_com"][trainer_id].send(
							(encrypted_data, len(feature_names) - i, True))
						logger.info(
							"trainer {} encrypted {} features to trainer {}.".format(
								self.node_id, len(feature_names), trainer_id))
					else:
						self.channels["trainer_com"][trainer_id].send(
							(encrypted_data, self.num_embed, False))
						logger.info(
							"trainer {} encrypted {} features to trainer {}.".format(
								self.node_id, i + self.num_embed, trainer_id))
					self.channels["trainer_com"][trainer_id].recv()

		self.channels["trainer_corr_com"].send(self._summary)
		self.save()

	def save(self):
		save_dir = str(Path(self.output.get("path")))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		model_name = self.output.get("corr")["name"]
		model_path = Path(save_dir, model_name)
		with open(model_path, 'wb') as f:
			pickle.dump(self._local_summary, f)
		logger.info("model saved as: {}.".format(model_path))
