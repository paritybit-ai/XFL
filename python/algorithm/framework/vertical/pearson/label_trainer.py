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

from common.checker.matcher import get_matched_config
from common.checker.x_types import All
from service.fed_config import FedConfig
from service.fed_node import FedNode
from service.fed_control import _update_progress_finish
from .base import VerticalPearsonBase
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.utils.logger import logger
from algorithm.core.encryption_param import PlainParam, PaillierParam
from common.crypto.paillier.paillier import Paillier
from algorithm.core.paillier_acceleration import embed, umbed


class VerticalPearsonLabelTrainer(VerticalPearsonBase):
	def __init__(self, train_conf: dict):
		"""

		Args:
			train_conf:
		"""
		self.sync_channel = BroadcastChannel(name="sync")
		self._sync_config(train_conf)
		super().__init__(train_conf, label=False)
		self.feature_mapping = dict()
		self.channels = dict()
		self.channels["encryption_context"] = BroadcastChannel(name="encryption_context")
		self.channels["sample_idx_com"] = BroadcastChannel(name="sample_idx_com")
		self.channels["trainer_corr_com"] = dict()
		self.channels["trainer_feature_com"] = dict()
		self.node_id = FedNode.node_id
		for party_id in FedConfig.get_trainer():
			self.channels["trainer_feature_com"][party_id] = DualChannel(
				name="trainer_feature_com_" + party_id, ids=[self.node_id, party_id]
			)
		for party_id in FedConfig.get_trainer():
			self.channels["trainer_corr_com"][party_id] = DualChannel(
				name="trainer_corr_com_" + party_id, ids=[self.node_id, party_id]
			)

		if isinstance(self.encryption_param, (PlainParam, type(None))):
			self.private_context = None
		elif isinstance(self.encryption_param, PaillierParam):
			self.private_context = Paillier.context(
				self.encryption_param.key_bit_size,
				self.encryption_param.djn_on
			)
			self.public_context = self.private_context.to_public()
			self.channels["encryption_context"].broadcast(self.public_context.serialize(), use_pickle=False)
		else:
			raise TypeError(f"Encryption param type {type(self.encryption_param)} not valid.")

	def _sync_config(self, config):
		sync_rule = {
			"train_info": All()
		}
		config_to_sync = get_matched_config(config, sync_rule)
		self.sync_channel.broadcast(config_to_sync)

	def fit(self):
		logger.info("vertical pearson label trainer start.")
		data = self._select_columns()
		if self.sample_size is not None and self.sample_size < len(self.train_ids):
			logger.info("sampled %d data." % self.sample_size)
			sample_ids = np.random.choice(np.arange(len(self.train_ids)), self.sample_size)
			self.channels["sample_idx_com"].broadcast(sample_ids)
			data = data.iloc[sample_ids]

		data = data.apply(self.standardize)
		n = len(data)
		feature_names = data.columns.to_list()
		self.feature_mapping = self.string_encryption(feature_names)
		local_corr = np.dot(data.T, data)
		local_corr /= n
		self._summary["corr"] = dict()
		self._summary["corr"][(self.node_id, self.node_id)] = local_corr
		self._summary["features"] = {
			self.node_id: feature_names
		}
		self._summary["num_features"] = {
			self.node_id: len(feature_names)
		}

		remote_corr = dict()
		for party_id in FedConfig.get_trainer():
			remote_corr[party_id] = []
			if len(feature_names):
				self.channels["trainer_feature_com"][party_id].send(False)
			else:
				self.channels["trainer_feature_com"][party_id].send(True)

		if isinstance(self.encryption_param, (PlainParam, type(None))):
			for idx, f in enumerate(feature_names):
				for party_id in FedConfig.get_trainer():
					if f != feature_names[-1]:
						self.channels["trainer_feature_com"][party_id].send(
							(data[f].to_numpy(), False))
					else:
						self.channels["trainer_feature_com"][party_id].send(
							(data[f].to_numpy(), True))
				logger.info("label trainer encrypted %d features." % (idx + 1))
				for party_id in FedConfig.get_trainer():
					corr_mat = self.channels["trainer_feature_com"][party_id].recv()
					remote_corr[party_id].append(corr_mat)
			for party_id in FedConfig.get_trainer():
				remote_corr[party_id] = np.array(remote_corr[party_id]) / n
		if isinstance(self.encryption_param, PaillierParam):
			for i in range(0, len(feature_names), self.num_embed):
				embedded_data = embed(data.iloc[:, i:(i + self.num_embed)].to_numpy().T)
				encrypted_data = Paillier.encrypt(
					context=self.private_context,
					data=embedded_data,
					obfuscation=True,
					num_cores=self.max_num_cores
				)
				for party_id in FedConfig.get_trainer():
					if i + self.num_embed >= len(feature_names):
						self.channels["trainer_feature_com"][party_id].send(
							(encrypted_data, True))
						logger.info("label trainer encrypted %d features." % len(feature_names))
					else:
						self.channels["trainer_feature_com"][party_id].send(
							(encrypted_data, False))
						logger.info("label trainer encrypted %d features." % (i + self.num_embed))
				for party_id in FedConfig.get_trainer():
					emb_enc_corr_mat = self.channels["trainer_feature_com"][party_id].recv()
					emb_corr_mat = Paillier.decrypt(
						self.private_context, emb_enc_corr_mat, num_cores=self.max_num_cores, out_origin=True
					)
					result = []
					if i + self.num_embed >= len(feature_names):
						umbed_num = len(feature_names) - i
					else:
						umbed_num = self.num_embed
					for r in umbed(emb_corr_mat, umbed_num):
						result.append(r)
					remote_corr[party_id].append(np.array(result) / (10 ** self.encryption_param.precision))
			for party_id in FedConfig.get_trainer():
				if remote_corr[party_id]:
					remote_corr[party_id] = np.concatenate(remote_corr[party_id]) / n

		logger.info("label trainer encrypted all features.")
		for party_id in FedConfig.get_trainer():
			remote_features = self.channels["trainer_feature_com"][party_id].recv()
			self._summary["features"][party_id] = remote_features
			self._summary["num_features"][party_id] = len(remote_features)
			self._summary["corr"][(self.node_id, party_id)] = remote_corr[party_id]

		logger.info("label trainer get remote correlation matrices.")

		logger.info("get correlation matrix between trainers.")
		for party_id in FedConfig.get_trainer():
			other_summary = self.channels["trainer_corr_com"][party_id].recv()
			for k, v in other_summary.items():
				if k[0] != k[1]:
					if self.encryption == "plain":
						self._summary["corr"][k] = v[0].T / n
					elif self.encryption == "paillier":
						remote_corr = []
						for emb_enc_corr_mat, pack_num in zip(v[0], v[1]):
							emb_corr_mat = Paillier.decrypt(
								self.private_context, emb_enc_corr_mat, num_cores=self.max_num_cores, out_origin=True
							)
							corr_mat = []
							for r in umbed(emb_corr_mat, pack_num):
								corr_mat.append(r)
							corr_mat = np.array(corr_mat) / (10 ** self.encryption_param.precision)
							remote_corr.append(corr_mat)
						if remote_corr:
							self._summary["corr"][k] = np.concatenate(remote_corr).T / n
						else:
							self._summary["corr"][k] = []
				else:
					self._summary["corr"][k] = v

        # update the progress of 100 to show the training is finished
		_update_progress_finish()
		self.save()

	def save(self):
		save_dir = str(Path(self.output.get("path")))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		model_name = self.output.get("corr")["name"]
		model_path = Path(save_dir, model_name)
		summary = self.merge_summary()
		with open(model_path, 'wb') as f:
			pickle.dump(summary, f)
		logger.info("model saved as: {}.".format(model_path))

	def merge_summary(self):
		parties = FedConfig.get_label_trainer() + FedConfig.get_trainer()
		cor_mat = []
		features = []
		sources = []
		num_feature = 0
		for i in range(len(parties)):
			row = []
			for j in range(len(parties)):
				if (parties[i], parties[j]) in self._summary["corr"]:
					corr_mat = self._summary["corr"][(parties[i], parties[j])]
					if min(np.array(corr_mat).shape) > 0:
						row.append(corr_mat)
					else:
						row.append(np.zeros(
							(self._summary["num_features"][parties[i]], self._summary["num_features"][parties[j]])))
				else:
					corr_mat = self._summary["corr"][(parties[j], parties[i])]
					if min(np.array(corr_mat).shape) > 0:
						row.append(corr_mat.T)
					else:
						row.append(np.zeros(
							(self._summary["num_features"][parties[i]], self._summary["num_features"][parties[j]])))
			cor_mat.append(np.concatenate(row, axis=1))
			features.extend(self._summary["features"][parties[i]])
			sources.extend([parties[i]] * self._summary["num_features"][parties[i]])
			num_feature += self._summary["num_features"][parties[i]]
		cor_mat = np.concatenate(cor_mat, axis=0)
		ret = {
			"corr": cor_mat,
			"features": features,
			"feature_source": sources,
			"num_features": self._summary["num_features"]
		}
		return ret
