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

import pandas as pd

from common.communication.gRPC.python.channel import BroadcastChannel
from common.utils.logger import logger
from .base import VerticalFeatureSelectionBase


class VerticalFeatureSelectionTrainer(VerticalFeatureSelectionBase):
	def __init__(self, train_conf: dict):
		"""

		Args:
			train_conf:

		"""
		super().__init__(train_conf, label=False)
		self.channels = dict()
		self.channels["feature_id_com"] = BroadcastChannel(name="feature_id_com")
		self.feature_mapping = dict()

	def _common_filter(self, params):
		metrics = params.get("metrics", 'iv')
		if isinstance(metrics, str):
			metrics = [metrics]
		elif isinstance(params.get("metrics"), list):
			pass
		else:
			raise NotImplementedError("param metrics must be a string or a list.")
		for metric in metrics:
			if metric == "iv":
				for k, v in self.iv_result["feature_mapping"].items():
					self.feature_mapping[k] = v

	def _correlation_filter(self):
		feature_id_list = self.channels["feature_id_com"].recv()
		mapping = {}
		reversed_feature_mapping = dict()
		for k, v in self.feature_mapping.items():
			reversed_feature_mapping[v] = k

		for feature_id in feature_id_list:
			if feature_id in reversed_feature_mapping:
				local_feature_name = reversed_feature_mapping.get(feature_id)
				if local_feature_name in self.corr_result["feature_mapping"]:
					mapping[feature_id] = self.corr_result["feature_mapping"][local_feature_name]
			else:
				continue
		self.channels["feature_id_com"].send(mapping)
		self.feature_mapping = self.corr_result["feature_mapping"]

	def fit(self):
		logger.info("feature selection trainer start.")
		for k, v in self.filter_params.items():
			if k == "common":
				self._common_filter(v)
			elif k == "correlation":
				self._correlation_filter()
			else:
				raise NotImplementedError("method {} is not implemented.".format(k))
		remain_id_list = self.channels["feature_id_com"].recv()
		res = dict()
		for k, v in self.feature_mapping.items():
			if v in remain_id_list:
				res[k] = v
		self.feature_mapping = res
		self.save()
		self.transform()

	def transform(self):
		if not self.transform_stages:
			return None
		selected_features = [_ for _ in self.feature_mapping]
		if "train" in self.transform_stages:
			self.train_features = self.train_features[selected_features]
			if self.train_label is not None:
				df = pd.concat([self.train_label, self.train_features], axis=1).set_index(self.train_id)
			else:
				df = self.train_features.set_index(self.train_id)
			df.to_csv(
				Path(self.output["trainset"].get("path"), self.output["trainset"].get("name")),
				header=True, index=True, index_label="id", float_format="%.6g"
			)
		if "valid" in self.transform_stages:
			self.val_features = self.val_features[selected_features]
			if self.val_label is not None:
				df = pd.concat([self.val_label, self.val_features], axis=1).set_index(self.val_id)
			else:
				df = self.val_features.set_index(self.val_id)
			df.to_csv(
				Path(self.output["valset"].get("path"), self.output["valset"].get("name")),
				header=True, index=True, index_label="id", float_format="%.6g"
			)

	def save(self):
		save_dir = str(Path(self.output.get("model")["path"]))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		model_name = self.output.get("model")["name"]
		model_path = Path(save_dir, model_name)

		features = list(self.feature_mapping.keys())

		with open(model_path, 'wb') as f:
			pickle.dump(self.feature_mapping, f)
		logger.info("model saved as: {}.".format(model_path))

		self._rewrite_model(features)
