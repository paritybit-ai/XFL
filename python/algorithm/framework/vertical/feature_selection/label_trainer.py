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

from service.fed_control import ProgressCalculator
from common.checker.matcher import get_matched_config
from common.checker.x_types import All
from common.communication.gRPC.python.channel import BroadcastChannel
from common.utils.logger import logger
from .base import VerticalFeatureSelectionBase


class VerticalFeatureSelectionLabelTrainer(VerticalFeatureSelectionBase):
	def __init__(self, train_conf: dict):
		"""

		Args:
			train_conf:

		"""
		self.sync_channel = BroadcastChannel(name="sync")
		self._sync_config(train_conf)
		super().__init__(train_conf, label=True)
		self.progress_calculator = ProgressCalculator(len(self.filter))
		self.channels = dict()
		self.channels["feature_id_com"] = BroadcastChannel(name="feature_id_com")

	def _common_filter(self, params):
		logger.info("common_filter")
		metrics = params.get("metrics", 'iv')
		filter_method = params.get("filter_method", "threshold")
		if isinstance(metrics, str):
			metrics = [metrics]
		elif isinstance(params.get("metrics"), list):
			pass
		else:
			raise NotImplementedError("param metrics must be a string or a list.")
		for metric in metrics:
			if filter_method == "threshold":
				self.feature_info = self._threshold_filter(
					metric,
					params.get("threshold", 1)
				)

	def _sync_config(self, config):
		sync_rule = {
			"train_info": All()
		}
		config_to_sync = get_matched_config(config, sync_rule)
		self.sync_channel.broadcast(config_to_sync)

	def _threshold_filter(self, metric, threshold):
		ret = []
		if metric == "iv":
			for r in self.feature_info:
				if r["iv"] < threshold:
					logger.info("filter feature {} < threshold {}".format(r["feature_id"], threshold))
					continue
				ret.append(r)
		else:
			raise NotImplementedError("metric {} is not supported".format(metric))
		return ret

	def _correlation_filter(self, params):
		logger.info("correlation_filter")
		sort_metric = params.get("sort_metric", 'iv')
		correlation_threshold = params.get("correlation_threshold", 0.1)
		self.channels["feature_id_com"].broadcast([_["feature_id"] for _ in self.feature_info])
		mapping = {}
		for d in self.channels["feature_id_com"].collect():
			for k, v in d.items():
				mapping[k] = v
		corr = self.corr_result["corr"]
		features = self.corr_result["features"]

		res = []
		filtered_features = []
		self.feature_info.sort(key=lambda x: x[sort_metric], reverse=True)
		for r in self.feature_info:
			f = mapping.get(r["feature_id"], r["feature_id"])
			if f in features:
				i = features.index(f)
			else:
				continue
			if features[i] in filtered_features:
				continue
			for f, s in zip(features, corr[i]):
				if f == features[i]:
					continue
				elif f in filtered_features:
					continue
				elif abs(s) > correlation_threshold:
					logger.info(
						"current feature {}, filtered feature {}.".format(
							features[i], f
						)
					)
					filtered_features.append(f)
			res.append({
				"feature_id": mapping.get(r["feature_id"], r["feature_id"]),
				"iv": r["iv"]
			})
		self.feature_info = res

	def fit(self):
		logger.info("feature selection label trainer start.")

		# iter_ is used to calculate the progress of the training
		iter_ = 0
		for k, v in self.filter.items():
			iter_ += 1
			if k == "common":
				self._common_filter(v)
			elif k == "correlation":
				self._correlation_filter(v)
			else:
				raise NotImplementedError("method {} is not implemented.".format(k))
			
			# calculate and update the progress of the training
			self.progress_calculator.cal_custom_progress(iter_)
		self.channels["feature_id_com"].broadcast([_["feature_id"] for _ in self.feature_info])
		self.save()
		self.transform()
		ProgressCalculator.finish_progress()

	def transform(self):
		if not self.transform_stages:
			return None
		selected_features = [_["feature_id"] for _ in self.feature_info]
		selected_features = [_ for _ in self.train_features.columns if _ in selected_features]
		if "train" in self.transform_stages:
			self.train_features = self.train_features[selected_features]
			if self.train_label is not None:
				df = pd.concat([self.train_label, self.train_features], axis=1).set_index(self.train_id)
			else:
				df = self.train_features.set_index(self.train_id)

			output_train_path = Path(self.output.get("path"), self.output["trainset"].get("name"))
			if not os.path.exists(os.path.dirname(output_train_path)):
				os.makedirs(os.path.dirname(output_train_path))
			df.to_csv(output_train_path, header=True, index=True, index_label="id", float_format="%.6g")
		if "valid" in self.transform_stages:
			self.val_features = self.val_features[selected_features]
			if self.val_label is not None:
				df = pd.concat([self.val_label, self.val_features], axis=1).set_index(self.val_id)
			else:
				df = self.val_features.set_index(self.val_id)
			output_val_path = Path(self.output.get("path"), self.output["valset"].get("name"))
			if not os.path.exists(os.path.dirname(output_val_path)):
				os.makedirs(os.path.dirname(output_val_path))
			df.to_csv(output_val_path, header=True, index=True, index_label="id", float_format="%.6g")

	def save(self):
		save_dir = str(Path(self.output.get("path")))
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		model_name = self.output.get("model")["name"]
		model_path = Path(save_dir, model_name)

		features = [_["feature_id"] for _ in self.feature_info]

		output = {
			"features": features,
			"num_of_features": len(self.feature_info)
		}

		with open(model_path, 'wb') as f:
			pickle.dump(output, f)
		logger.info("model saved as: {}.".format(model_path))

		self._rewrite_model(features)
