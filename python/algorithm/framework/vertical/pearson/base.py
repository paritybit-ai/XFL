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


import random
import string
from pathlib import Path
from typing import List

import pandas as pd

from algorithm.core.encryption_param import PlainParam, get_encryption_param
from common.crypto.paillier.utils import get_core_num
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger


class VerticalPearsonBase(TrainConfigParser):
	def __init__(self, train_conf: dict, label: bool = False):
		super().__init__(train_conf)
		self.label = label
		self.max_num_cores = 1
		self.sample_size = None
		self.encryption = "plain"
		self.encryption_param = PlainParam()
		self._init_config()
		self._init_data()
		self._local_summary = {}
		self._summary = {}
		self.batch_size = 2048
		self.num_embed = 10

	def _init_data(self):
		logger.info("init data loader.")
		if not self.input_trainset:
			return None
		input_info = self.input_trainset[0]
		file_path = str(Path(input_info.get("path"), input_info.get("name")))
		type_ = input_info.get("type", "None")
		if input_info.get("has_id", True):
			self.index_col = input_info.get("index_col", 'id')
		else:
			self.index_col = None
		if input_info.get("has_label", False):
			label_name = input_info.get("label_name", 'y')
			self.label = True
		else:
			label_name = None
		if type_ == "csv":
			if self.computing_engine == "local":
				df = pd.read_csv(file_path, index_col=self.index_col)
			# elif self.computing_engine == "spark":
			# 	df = ps.read_csv(file_path, index_col=self.index_col)
			else:
				raise NotImplementedError("Computing engine {} is not supported.".format(self.computing_engine))
		else:
			raise NotImplementedError("Dataset type {} is not supported.".format(type_))
		if self.label:
			feature_cols = [_ for _ in df.columns if _ != label_name]
			self.train_features = df[feature_cols]
			if label_name:
				self.train_label = df[label_name]
			else:
				self.train_label = None
		else:
			self.train_features = df
		self.train_ids = df.index

	def _init_config(self):
		params = self.train_info.get("params")
		self.column_indexes = params.get("column_indexes", -1)
		self.column_names = params.get("column_names", '')
		encryption_params = params.get("encryption_params", {"plain": {}})
		self.batch_size = params.get("batch_size", 2048)
		self.encryption = list(encryption_params.keys())[0]
		encryption_param = encryption_params[self.encryption]
		self.encryption_param = get_encryption_param(self.encryption, encryption_param)
		self.sample_size = params.get("sample_size", None)
		if self.encryption == "paillier" and self.encryption_param.parallelize_on:
			self.max_num_cores = get_core_num(params.get("max_cores", 999))
		else:
			self.max_num_cores = 1

	def _select_columns(self):
		if self.column_indexes == -1:
			return self.train_features
		elif isinstance(self.column_indexes, list):
			feature_start_index = 0
			if self.index_col is not None:
				feature_start_index += 1
			if self.label:
				feature_start_index += 1
			feature_names = self.train_features.columns.to_list()
			select_feature_cols = [feature_names[_ - feature_start_index] for _ in self.column_indexes]
			if self.column_names:
				for f in self.column_names.split(','):
					if f not in select_feature_cols:
						select_feature_cols.append(f)

			select_feature_cols.sort(key=lambda d: feature_names.index(d))
			return self.train_features[select_feature_cols]
		else:
			raise ValueError("column_indexes must be -1 or a list of int.")

	@staticmethod
	def standardize(x):
		mu = x.mean()
		sigma = x.std()
		if sigma > 0:
			return (x - mu) / sigma
		else:
			return x - mu

	@staticmethod
	def string_encryption(str_list: List[str]):
		ret = {}
		for s in str_list:
			ret[s] = ''.join(random.sample(string.ascii_letters + string.digits, 16))
		return ret
