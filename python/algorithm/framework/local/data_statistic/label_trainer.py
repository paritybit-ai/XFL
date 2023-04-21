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
import os
from pathlib import Path

import numpy as np
import pandas as pd
from service.fed_control import _update_progress_finish
from algorithm.core.data_io import CsvReader
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger


def float_transform(data):
    if isinstance(data, pd.Series):
        return data.apply(lambda x: float("%.6f" % x))
    elif isinstance(data, pd.DataFrame):
        col_name = data.columns[0]
        return pd.DataFrame(data[col_name].apply(lambda x: float("%.6f" % x)))


class LocalDataStatisticLabelTrainer:
    def __init__(self, train_conf):
        """
        support data statistic:
        If more than one file, their header should be the same.
        If there are missing values, they will be dropped before data statistic
        quantile: list of float, if not set, quartile will be calculated
        """
        # input config
        self.config = TrainConfigParser(train_conf)
        self.input_data = self.config.input.get("dataset", [])
        self.missing_value = [np.NaN, "", None, " ", "nan", "none", "null", "na", "None"]
        if self.input_data:
            if len(self.input_data) == 1:
                self.input_data_path = self.input_data[0].get("path")
                self.input_data_name = self.input_data[0].get("name")
                self.input_data_id = self.input_data[0].get("has_id", False)
                self.input_data_label = self.input_data[0].get("has_label", False)
                data_reader = CsvReader(path=os.path.join(self.input_data_path, self.input_data_name),
                                        has_id=self.input_data_id, has_label=self.input_data_label)
                self.data = data_reader.table.set_index(data_reader.ids)
            else:
                self.data = pd.DataFrame()
                for dataset_conf in self.input_data:
                    input_data_path = dataset_conf.get("path")
                    input_data_name = dataset_conf.get("name")
                    input_data_id = dataset_conf.get("has_id", False)
                    self.input_data_label = dataset_conf.get("has_label", False)
                    data_reader = CsvReader(path=os.path.join(input_data_path, input_data_name),
                                            has_id=input_data_id, has_label=self.input_data_label)
                    data = data_reader.table.set_index(data_reader.ids)
                    self.data = pd.concat([self.data, data])
        # drop label
        if self.input_data_label:
            self.y = pd.DataFrame(self.data.iloc[:, 0])
            self.data = self.data.iloc[:, 1:]

        # output config
        self.output_flag = self.config.output.get("summary", None)
        if self.output_flag is not None:
            self.output_path = self.config.output["path"]
            self.output_name = self.config.output["summary"]["name"]
            self.output_path_name = Path(self.output_path, self.output_name)
            if not os.path.exists(Path(self.output_path)):
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
        # init summary dict
        self.summary_dict = {}
        self.indicators = ["mean", "median", "missing_ratio", "min", "max", "variance", "std", "quantile",
                           "skewness", "kurtosis"]
        for i in self.indicators:
            self.summary_dict[i] = {}
        # missing value flag
        self.missing_flag = dict(zip(self.data.columns, [False] * len(self.data.columns)))
        # quantile config
        self.quantile = self.config.train_params.get("quantile", [0.25, 0.5, 0.75])

    def data_overview(self):
        data_shape = np.shape(self.data)
        self.summary_dict.update({"row_num": data_shape[0]})
        self.summary_dict.update({"column_num": data_shape[1]})
        self.summary_dict.update({"feature_names": list(self.data.columns)})

        logger.info("The shape of input data is {}*{}".format(data_shape[0], data_shape[1]))

    def missing_overview(self):

        def missing_count(feat):
            tmp = np.sum(self.data[feat].isin(self.missing_value))
            if tmp > 0:
                self.missing_flag[feat] = True
            self.summary_dict["missing_ratio"][feat] = float("%.6f" % (tmp / self.summary_dict["row_num"]))
            # replace all missing values to np.NaN
            self.data[feat] = self.data[feat].replace(self.missing_value, np.NaN)

        pd.Series(self.data.columns).apply(lambda x: missing_count(x))

    def label_overview(self):
        if self.input_data_label:
            label_name = self.y.columns[0]
            self.summary_dict.update({"label_num": self.y.groupby(label_name)[label_name].count().to_dict()})

    def get_mean(self, df):
        self.summary_dict["mean"].update(float_transform(df.mean()).to_dict())

    def get_median(self, df):
        self.summary_dict["median"].update(float_transform(df.median()).to_dict())

    def get_min_max(self, df):
        self.summary_dict["min"].update(float_transform(df.min()).to_dict())
        self.summary_dict["max"].update(float_transform(df.max()).to_dict())

    def get_variance(self, df):
        self.summary_dict["variance"].update(float_transform(df.var()).to_dict())

    def get_std(self, df):
        self.summary_dict["std"].update(float_transform(df.std()).to_dict())

    def get_quantile(self, df):
        self.summary_dict["quantile"].update(float_transform(df.quantile(self.quantile)).to_dict())

    def get_skewness(self, df):
        self.summary_dict["skewness"].update(float_transform(df.skew()).to_dict())

    def get_kurtosis(self, df):
        self.summary_dict["kurtosis"].update(float_transform(df.kurtosis()).to_dict())

    def fit(self):
        self.data_overview()
        self.missing_overview()
        self.label_overview()

        def feat_handle(feat):
            if self.missing_flag[feat]:
                data = pd.DataFrame(self.data[feat].dropna().apply(lambda x: eval(x)))
            else:
                data = self.data[[feat]]
            return data

        def feat_statistic(feat):
            feat_ = feat_handle(feat)
            self.get_mean(feat_)
            self.get_median(feat_)
            self.get_min_max(feat_)
            self.get_variance(feat_)
            self.get_std(feat_)
            self.get_quantile(feat_)
            self.get_skewness(feat_)
            self.get_kurtosis(feat_)
            logger.info("Feature {} calculated!".format(feat))

        pd.Series(self.data.columns).apply(lambda x: feat_statistic(x))
        # save
        if self.output_flag is not None:
            with open(self.output_path_name, "w") as wf:
                json.dump(self.summary_dict, wf)
        
        _update_progress_finish()
