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


import copy
import functools
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.pandas as ps
import torch

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger


class VerticalKmeansBase(TrainConfigParser):
	def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
		"""
		init

		Args:
			train_conf:
			label:
			*args:
			**kwargs:
		"""
		super().__init__(train_conf)
		self.k = 0
		self.max_iter = 0
		self.tol = 0.0
		self.is_converged = False
		self.encryption = "plain"
		self.label = label
		self.cluster_centers = []
		self.cluster_count_list = []
		self.train_features, self.train_label, self.train_ids = None, None, None
		self._init_data()
		self._init_config()

	def _init_data(self):
		logger.info("init data loader.")
		if not self.input_trainset:
			return None
		input_info = self.input_trainset[0]
		file_path = str(Path(input_info.get("path"), input_info.get("name")))
		type_ = input_info.get("type", "None")
		if input_info.get("has_id", True):
			index_col = input_info.get("index_col", 'id')
		else:
			index_col = None
		if input_info.get("has_label", True):
			label_name = input_info.get("label_name", 'y')
			self.label = True
		else:
			label_name = None
			self.label = False
		if type_ == "csv":
			if self.computing_engine == "local":
				df = pd.read_csv(file_path, index_col=index_col)
			elif self.computing_engine == "spark":
				df = ps.read_csv(file_path, index_col=index_col)
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
		"""
		Initialize model parameters

		Returns:

		"""
		params = self.train_info.get("params")
		self.k = params.get("k", 5)
		self.max_iter = params.get("max_iter", 20)
		self.tol = params.get("tol", 1e-5)
		self.random_seed = params.get("extra_config", {}).get("shuffle_seed", 2022)
		if self.identity != "assist_trainer":
			self._check()
		self.encryption = self.train_info["aggregation_config"]["encryption"]

	def _check(self):
		"""
		Check data and parameters

		Returns:

		"""
		if len(self.train_features) <= 0:
			raise ValueError("error: empty dataset.")
		if self.k < 2:
			raise ValueError("k must be an integer value larger than 1.")
		elif self.k > len(self.train_features):
			raise ValueError("k is larger than the size of current data.")

	@staticmethod
	def euclid_distance(u, center_list):
		result = []
		for i in range(len(center_list)):
			result.append(sum(np.square(center_list[i] - u)))
		return result

	def distance_table(self, centers):
		"""

		Args:
			centers: slices of the features

		Returns:

		"""
		if isinstance(centers, ps.DataFrame):
			centers = centers.to_numpy()
		elif isinstance(centers, pd.DataFrame):
			centers = centers.to_numpy()
		elif isinstance(centers, list):
			centers = np.array(centers)
		n = len(self.train_features)
		if self.train_features.empty:
			return
		d = functools.partial(self.euclid_distance, center_list=centers)
		dt = self.train_features.apply(d, axis=1)
		return torch.Tensor(list(chain.from_iterable(dt.to_numpy()))).reshape(n, self.k)

	@staticmethod
	def distance_between_centers(center_list):
		cluster_dist_list = []
		for i in range(0, len(center_list)):
			for j in range(0, len(center_list)):
				if j != i:
					cluster_dist_list.append(np.sum((np.array(center_list[i]) - np.array(center_list[j])) ** 2))
		return torch.Tensor(cluster_dist_list)

	def calc_centers(self, centers, cluster_result):
		"""
		Update cluster centers based on clustering results

		Args:
			centers: current center slice
			cluster_result: result of clustering labels

		Returns:

		"""
		feature_sum = {}
		feature_count = {}
		for feature, label in zip(self.train_features.values, cluster_result):
			if label not in feature_sum:
				feature_sum[label] = copy.deepcopy(feature)
			else:
				feature_sum[label] += feature
			feature_count[label] = feature_count.get(label, 0) + 1
		center_list = []
		# for k in centroid_feature_sum:
		for k in range(self.k):
			if k not in feature_sum:
				if isinstance(centers, ps.DataFrame):
					center_list.append(centers.iloc[k])
				elif isinstance(centers, pd.DataFrame):
					center_list.append(centers.iloc[k])
				elif isinstance(centers, list):
					center_list.append(centers[k])
				else:
					raise NotImplementedError
			else:
				count = feature_count[k]
				center_list.append(feature_sum[k] / count)
		return center_list

	def calc_cluster_count(self, cluster_result):
		"""

		Args:
			cluster_result: result of clustering labels

		Returns:

		"""
		feature_count = {}
		for label in cluster_result:
			feature_count[label] = feature_count.get(label, 0) + 1
		cluster_count_list = []
		count_all = len(cluster_result)
		for k in range(self.k):
			if k not in feature_count:
				cluster_count_list.append([k, 0, 0])
			else:
				count = feature_count[k]
				cluster_count_list.append([k, count, count / count_all])
		return cluster_count_list

	@staticmethod
	def calc_tolerance(centers, centers_new):
		"""
		Calculate convergence metrics

		Returns:

		"""
		if isinstance(centers, ps.DataFrame):
			centers = centers.to_numpy()
		elif isinstance(centers, pd.DataFrame):
			centers = centers.to_numpy()
		elif isinstance(centers, list):
			centers = np.array(centers)
		return np.sum(np.sum((centers - np.array(centers_new)) ** 2, axis=1))
