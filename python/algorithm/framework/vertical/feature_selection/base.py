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


import json
import pickle
from pathlib import Path

import pandas as pd

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from common.model.python.feature_model_pb2 import WOEModel
from google.protobuf import json_format
from service.fed_job import FedJob
from service.fed_node import FedNode


class VerticalFeatureSelectionBase(TrainConfigParser):
	def __init__(self, train_conf: dict, label: bool = False):
		super().__init__(train_conf)
		self.label = label
		self.feature_info = []
		self.iv_result = None
		self.corr_result = None
		self.transform_stages = []
		self.train_id, self.train_label, self.train_features = None, None, None
		self.val_id, self.val_label, self.val_features = None, None, None
		self._init_config()
		self._load_feature_info()
		self._init_data()

	def _init_data(self):
		logger.info("init data loader.")
		if not self.input_trainset:
			return None
		self.train_id, self.train_label, self.train_features = self._load_data(self.input_trainset[0])
		self.transform_stages.append("train")
		if not self.input_valset:
			return None
		self.val_id, self.val_label, self.val_features = self._load_data(self.input_valset[0])
		self.transform_stages.append("valid")

	def _load_data(self, input_info):
		file_path = str(Path(input_info.get("path"), input_info.get("name")))
		type_ = input_info.get("type", "None")
		if input_info.get("has_id", True):
			index_col = input_info.get("index_col", 'id')
		else:
			index_col = None
		if input_info.get("has_label", False):
			label_name = input_info.get("label_name", 'y')
			self.label = True
		else:
			label_name = None
		if type_ == "csv":
			if self.computing_engine == "local":
				df = pd.read_csv(file_path, index_col=index_col)
			# elif self.computing_engine == "spark":
			# 	df = ps.read_csv(file_path, index_col=index_col)
			else:
				raise NotImplementedError("Computing engine {} is not supported.".format(self.computing_engine))
		else:
			raise NotImplementedError("Dataset type {} is not supported.".format(type_))
		if self.label:
			feature_cols = [_ for _ in df.columns if _ != label_name]
			features = df[feature_cols]
			if label_name:
				label = df[label_name]
			else:
				label = None
		else:
			features = df
			label = None
		ids = df.index
		return ids, label, features

	def _init_config(self):
		params = self.train_info.get("train_params")
		self.filter = params.get("filter", {})

	def _parse_from_iv(self, params):
		path = params["path"].replace("[JOB_ID]", str(FedJob.job_id)).replace("[NODE_ID]", str(FedNode.node_id))
		with open(Path(path, params["name"])) as f:
			res = json.load(f)
		if self.label:
			res = res.get("iv")
			for k, v in res.items():
				self.feature_info.append({
					"feature_id": k,
					"iv": v
				})
		return res

	@staticmethod
	def _parse_corr_result(params):
		path = params["path"].replace("[JOB_ID]", str(FedJob.job_id)).replace("[NODE_ID]", str(FedNode.node_id))
		with open(Path(path, params["name"]), 'rb') as f:
			ret = pickle.load(f)
		return ret

	def _load_feature_info(self):
		if "iv_result" in self.input:
			self.iv_result = self._parse_from_iv(self.input.get("iv_result"))
		if "corr_result" in self.input:
			self.corr_result = self._parse_corr_result(self.input.get("corr_result"))

	def _rewrite_model(self, features):
		if self.input.get("model"):
			params = self.input.get("model")
			path = params["path"].replace("[JOB_ID]", str(FedJob.job_id)).replace("[NODE_ID]", str(FedNode.node_id))
			file_path = Path(path, params["name"])
			with open(file_path, 'rb') as f:
				byte_str = f.read()
			woe = WOEModel()
			woe.ParseFromString(byte_str)
			d = json_format.MessageToDict(woe,
			                              including_default_value_fields=True,
			                              preserving_proto_field_name=True)
			ret = dict()
			for k, v in d["feature_binning"].items():
				if v["feature"] in features:
					ret[k] = v
			woe = WOEModel()
			json_format.ParseDict({"feature_binning": ret}, woe)
			with open(file_path, "wb") as f:
				f.write(woe.SerializeToString())
			logger.info("rewrite model in {}.".format(file_path))
