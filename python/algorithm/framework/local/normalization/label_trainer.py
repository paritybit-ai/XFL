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

import numpy as np
import pandas as pd
from scipy.linalg import norm
import torch

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from common.utils.utils import save_model_config


class LocalNormalizationLabelTrainer(TrainConfigParser):
	def __init__(self, train_conf):
		"""

		Args:
			train_conf:
		"""
		super().__init__(train_conf)
		self.train_data = None
		self.valid_data = None
		self.save_dir = None
		self.skip_cols = []
		self.transform_switch = False
		self._parse_config()
		self._init_data()
		self.export_conf = [{
			"class_name": "LocalNormalization",
			"filename": self.save_model_name
		}]

	def _parse_config(self) -> None:
		"""
		parse algo config

		Returns:
		"""
		self.save_dir = self.output.get("path")
		self.save_model_name = self.output.get("model", {}).get("name")
		self.save_trainset_name = self.output.get("trainset", {}).get("name")
		self.save_valset_name = self.output.get("valset", {}).get("name")

	def _init_data(self) -> None:
		"""
		load raw data
		1. using train set to generate the normalizer
		2. apply the normalizer to the valid set for subsequent model training

		Returns:
		"""
		if self.input_trainset:
			df_list = []
			for ts in self.input_trainset:
				if ts.get("type") == "csv":
					df_list.append(pd.read_csv(os.path.join(ts.get("path"), ts.get("name"))))
					if ts.get("has_id") and 'id' not in self.skip_cols:
						self.skip_cols.append('id')
					if ts.get("has_label") and 'y' not in self.skip_cols:
						self.skip_cols.append('y')
				else:
					raise NotImplementedError(
						"Load function {} does not Implemented.".format(ts.get("type"))
					)
			self.train_data = pd.concat(df_list)
			self.skip_cols.extend(self.train_data.columns[self.train_data.dtypes == 'object'])
			if len(self.skip_cols) > 0:
				logger.info("Skip columns {}".format(','.join(self.skip_cols)))

		if self.input_valset:
			df_list = []
			for vs in self.input_valset:
				df_list.append(pd.read_csv(os.path.join(vs.get("path"), vs.get("name"))))
				self.transform_switch = True
			self.valid_data = pd.concat(df_list)

	def fit(self) -> None:
		"""
		train a normalizer

		train_params:
		- axis = {1 if normalization is done by samples, 0 if normalization is done by feature}
		- norm = {"l1", "l2", "max"}

		output:
		- the .csv files which save the transformed data
		- the .pt file which saves the normalizer

		:return: None
		"""
		if self.train_data is None:
			logger.info("no data, skip stage.".format(self.identity))
			return
		normalizer_dict = {}
		cols = [_ for _ in self.train_data.columns if _ not in self.skip_cols]
		if self.train_params.get("axis") == 1:
			valid_normalizer = None
			# independently normalize each sample
			if self.train_params.get("norm") == "l1":
				train_normalizer = self.train_data[cols].apply(lambda x: norm(x, ord=1), axis=1)
				if self.transform_switch:
					valid_normalizer = self.valid_data[cols].apply(lambda x: norm(x, ord=1), axis=1)
			elif self.train_params.get("norm") == "l2":
				train_normalizer = self.train_data[cols].apply(lambda x: norm(x, ord=2), axis=1)
				if self.transform_switch:
					valid_normalizer = self.valid_data[cols].apply(lambda x: norm(x, ord=2), axis=1)
			elif self.train_params.get("norm") == "max":
				train_normalizer = self.train_data[cols].apply(lambda x: np.max(np.abs(x)), axis=1)
				if self.transform_switch:
					valid_normalizer = self.valid_data[cols].apply(lambda x: np.max(np.abs(x)), axis=1)
			else:
				raise NotImplementedError("norm {} is invalid.".format(self.train_params.get("norm", '')))
			train_normalizer = np.where(np.abs(train_normalizer - 0) < 1e-6, 1e-6, train_normalizer)
			if self.transform_switch:
				valid_normalizer = np.where(np.abs(valid_normalizer - 0) < 1e-6, 1e-6, valid_normalizer)
			for f in cols:
				self.train_data[f] /= train_normalizer
				if self.transform_switch:
					self.valid_data[f] /= valid_normalizer
			normalizer_dict["axis"] = 1
			normalizer_dict["norm"] = self.train_params["norm"]
		elif self.train_params.get("axis") == 0:
			# normalize each feature
			default_norm = self.train_params.get("norm")
			norm_dict = {}
			normalizers = {}
			if default_norm is None:
				pass
			elif default_norm not in ("l1", "l2", "max"):
				raise NotImplementedError("norm {} is invalid.".format(self.train_params.get("norm", '')))
			else:
				for f in cols:
					norm_dict[f] = default_norm
			for f in self.train_params.get("feature_norm", []):
				if self.train_params["feature_norm"][f].get("norm", default_norm) not in (
					"l1", "l2", "max"):
					raise NotImplementedError("norm {} is invalid.".format(self.train_params.get("norm", '')))
				elif f not in cols:
					raise KeyError("Feature {} cannot be found in df.".format(f))
				else:
					norm_dict[f] = self.train_params["feature_norm"][f]["norm"]
			for idx, (f, n) in enumerate(norm_dict.items()):
				logger.info("{}: Count={}, Min={}, Max={}, Unique={}.".format(
					f, self.train_data[f].count(), self.train_data[f].min(),
					self.train_data[f].max(), self.train_data[f].nunique()
				))
				if n == "l1":
					normalizer = norm(self.train_data[f].values, ord=1)
				elif n == "l2":
					normalizer = norm(self.train_data[f].values, ord=2)
				elif n == "max":
					normalizer = np.max(np.abs(self.train_data[f].values))
				else:
					normalizer = 1
				if np.abs(normalizer - 0) < 1e-6:
					normalizer = 1
				self.train_data[f] /= normalizer
				if self.transform_switch:
					self.valid_data[f] /= normalizer
				logger.info("{}: Norm={}.".format(f, normalizer))
				normalizers[idx] = {"feature": f, "norm_value": normalizer}
			normalizer_dict["axis"] = 0
			normalizer_dict["normalizer"] = normalizers
		elif "axis" in self.train_params:
			raise ValueError("axis {} is invalid.".format(self.train_params["axis"]))
		else:
			raise KeyError("cannot find the param axis, which is required for normalization.")

		self.save(normalizer_dict)

	def save(self, normalizer):
		if self.save_dir:
			self.save_dir = Path(self.save_dir)
		else:
			return

		save_model_config(stage_model_config=self.export_conf,
		                  save_path=self.save_dir)

		dump_path = self.save_dir / Path(self.save_model_name)
		torch.save(normalizer, dump_path)
		logger.info(
			"Normalize results saved as {}.".format(dump_path)
		)

		save_trainset_path = self.save_dir / Path(self.save_trainset_name)
		if not os.path.exists(os.path.dirname(save_trainset_path)):
			os.makedirs(os.path.dirname(save_trainset_path))
		self.train_data.to_csv(save_trainset_path, float_format='%.6g', index=False)
		logger.info("Data saved as {}, length: {}.".format(save_trainset_path, len(self.train_data)))

		if self.transform_switch:
			save_valset_path = self.save_dir / Path(self.save_valset_name)
			if not os.path.exists(os.path.dirname(save_valset_path)):
				os.makedirs(os.path.dirname(save_valset_path))
			self.valid_data.to_csv(save_valset_path, float_format='%.6g', index=False)
			logger.info("Data saved as {}, length: {}.".format(save_valset_path, len(self.valid_data)))

		logger.info("Data normalize completed.")
