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
from sklearn.preprocessing import OneHotEncoder
import torch

from algorithm.core.data_io import CsvReader
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from sklearn.impute import SimpleImputer

from common.utils.utils import save_model_config


def data_impute(form, strategy, fill=None):
    if strategy != 'constant':
        return SimpleImputer(missing_values=form, strategy=strategy, copy=False)
    else:
        return SimpleImputer(missing_values=form, strategy=strategy, fill_value=fill, copy=False)


def config_combination(config_a, config_b):
    if isinstance(config_a, list):
        if isinstance(config_b, list):
            config_combine = set(config_a + config_b)
        else:
            config_combine = set(config_a + [config_b])
    else:
        if isinstance(config_b, list):
            config_combine = set([config_a] + config_b)
        else:
            config_combine = set([config_a] + [config_b])
    if len(config_combine) > 1:
        return list(config_combine)
    elif len(config_combine) == 1:
        return list(config_combine)[0]
    else:
        return config_combine


class LocalFeaturePreprocessLabelTrainer(TrainConfigParser):
    def __init__(self, train_conf):
        """

        Args:
            train_conf:
        """
        super().__init__(train_conf)
        self.train = None
        self.val = None
        self.save_dir = None
        self.transform_switch = False
        self.impute_dict = {}
        self.outlier_dict = {}
        self.onehot_dict = {}
        self.imputer_values_overall = []
        self.imputer_strategy_overall = "mean"  # default
        self.imputer_fillvalue_overall = None  # default
        self.impute_dict = {}
        self.onehot_feat_conf = {}
        self.feature_flag = False  # indicating imputer by features whether to do
        self.model_file = {}
        self._init_data()
        self._parse_config()

    def _parse_config(self) -> None:
        """
        parse algo config
        missing_values: int, float, str or list, e.g. [-999, 999] or ["none", "null", "na", ""], default=null
        strategy: str, default="mean"
        fill_value: str or numerical value if strategy == "constant", default=None
        Returns:
        """
        self.save_model = self.output.get("model", {})
        if len(self.save_model) > 0:
            self.save_model_name = self.save_model.get("name")
            self.save_dir = self.save_model.get("path")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            model_name = self.output.get("model")["name"]
            self.save_model_path = Path(self.save_dir, model_name)
            self.export_conf = [{
                "class_name": "LocalFeaturePreprocess",
                "filename": self.save_model_name
            }]
        # missing config
        self.missing_conf = self.train_params.get("missing_params", {})
        if len(self.missing_conf) > 0:
            self.missing_values_overall = self.missing_conf.get(
                "missing_values", [np.NaN, '', None, ' ', 'nan', 'none', 'null', 'na', 'None'])
            # transform null: None to default missing_values config
            if self.missing_values_overall is None:
                self.missing_values_overall = [np.NaN, '', None, ' ', 'nan', 'none', 'null', 'na', 'None']
            self.missing_strategy_overall = self.missing_conf.get("strategy", "mean")
            self.missing_fillvalue_overall = self.missing_conf.get("fill_value", None)
            self.missing_feat_conf = self.missing_conf.get("missing_feat_params", {})
            self.imputer_values_overall = self.missing_values_overall
            self.imputer_strategy_overall = self.missing_strategy_overall
            self.imputer_fillvalue_overall = self.missing_fillvalue_overall
            logger.info("Missing values need to be imputed")
        # outlier config
        self.outlier_conf = self.train_params.get("outlier_params", {})
        if len(self.outlier_conf) > 0:
            self.outlier_values_overall = self.outlier_conf.get("outlier_values", [])
            self.outlier_feat_conf = self.outlier_conf.get("outlier_feat_params", {})
            self.imputer_values_overall = config_combination(self.imputer_values_overall, self.outlier_values_overall)
            logger.info("Outlier values need to be imputed")
        # initialize impute_dict
        if self.imputer_values_overall:
            self.impute_dict = dict(zip(self.columns, [{"missing_values": self.imputer_values_overall,
                                                        "strategy": self.imputer_strategy_overall,
                                                        "fill_value": self.imputer_fillvalue_overall}
                                                       for i in self.columns]))
        # if different features have different missing_values
        if len(self.missing_conf) > 0:
            if len(self.missing_feat_conf) > 0:
                for key in self.missing_feat_conf.keys():
                    if len(self.missing_feat_conf[key]) > 0:
                        missing_values_feat = self.missing_feat_conf[key].get("missing_values", None)
                        if missing_values_feat is not None:
                            self.impute_dict[key]["missing_values"] = missing_values_feat
                            self.feature_flag = True
                        missing_strategy_feat = self.missing_feat_conf[key].get("strategy", None)
                        if missing_strategy_feat is not None:
                            self.impute_dict[key]["strategy"] = missing_strategy_feat
                            self.feature_flag = True
                        missing_fillvalue_feat = self.missing_feat_conf[key].get("fill_value", None)
                        if missing_fillvalue_feat is not None:
                            self.impute_dict[key]["fill_value"] = missing_fillvalue_feat
                            self.feature_flag = True
        # if different features have different outlier_values
        if len(self.outlier_conf) > 0:
            if len(self.outlier_feat_conf) > 0:
                for key in self.outlier_feat_conf.keys():
                    if len(self.outlier_feat_conf[key]) > 0:
                        outlier_values_feat = self.outlier_feat_conf[key].get("outlier_values", None)
                        if outlier_values_feat is not None:
                            if key in self.impute_dict.keys():
                                self.impute_dict[key]["missing_values"] = config_combination(
                                    self.impute_dict[key]["missing_values"], outlier_values_feat)
                            else:
                                self.impute_dict[key] = {}
                                self.impute_dict[key]["missing_values"] = outlier_values_feat
                            self.feature_flag = True
        # check the three params
        if len(self.impute_dict) > 0:
            for key in self.impute_dict.keys():
                if "strategy" not in self.impute_dict[key].keys():
                    self.impute_dict[key]["strategy"] = self.imputer_strategy_overall
                    self.impute_dict[key]["fill_value"] = self.imputer_fillvalue_overall
        # onehot config
        self.onehot_conf = self.train_params.get("onehot_params", {})
        if len(self.onehot_conf) > 0:
            self.onehot_feat_conf = self.onehot_conf.get("feature_params", {})
        # output config
        self.save_trainset_name = self.output.get("trainset", {})
        self.save_valset_name = self.output.get("valset", {})

    def __load_data(self, config) -> CsvReader:
        if len(config) > 1:
            logger.warning("More than one dataset is not supported.")

        config = config[0]
        if config["type"] == "csv":
            data_reader = CsvReader(path=os.path.join(config["path"], config["name"]), has_id=config["has_id"],
                                    has_label=config["has_label"])
        else:
            raise NotImplementedError("Dataset type {} is not supported.".format(config["type"]))
        return data_reader

    def _init_data(self) -> None:
        if len(self.input["trainset"]) > 0:
            data: CsvReader = self.__load_data(self.input["trainset"])
            self.train = data.table.set_index(data.ids)
            self.label_name = data.label_name()
            if self.label_name is not None:
                self.train_label = self.train[[self.label_name]]
                self.train = self.train.drop(columns=self.label_name)
            self.columns = self.train.columns
            self.train_ids = data.ids
        else:
            raise NotImplementedError("Trainset was not configured.")

        if len(self.input["valset"]) > 0:
            data: CsvReader = self.__load_data(self.input["valset"])
            self.val = data.table.set_index(data.ids)
            if self.label_name is not None:
                self.val_label = self.val[[self.label_name]]
                self.val = self.val.drop(columns=self.label_name)
            self.val_ids = data.ids

    def impute(self):
        # fill missing_values for different features
        def imputer_series(data, col, flag):
            if flag == "train":
                missing_value_new = self.impute_dict[col]["missing_values"]
                if isinstance(missing_value_new, list) and len(missing_value_new) > 0:
                    data[col] = data[[col]].replace(self.impute_dict[col]["missing_values"], np.NaN)
                    missing_value_new = np.NaN
                imputer = data_impute(missing_value_new, self.impute_dict[col]["strategy"],
                                      self.impute_dict[col]["fill_value"])
                imputer.fit(data[[col]])
                data[col] = imputer.transform(data[[col]])
                imputer_list.update({col: imputer})
            elif flag == "val":
                if isinstance(self.impute_dict[col]["missing_values"], list) and \
                        len(self.impute_dict[col]["missing_values"]) > 0:
                    data[[col]] = data[[col]].replace(self.impute_dict[col]["missing_values"], np.NaN)
                data[col] = imputer_list[col].transform(data[[col]])

        if not self.feature_flag and len(self.imputer_values_overall) > 0:
            # if all features are imputed as a whole
            imputer_values_overall = self.imputer_values_overall
            # deal with more than one missing_values: transform the missing_values to np.NaN
            if isinstance(self.imputer_values_overall, list):
                self.train = self.train.replace(self.imputer_values_overall, np.NaN)
                if self.val is not None:
                    self.val = self.val.replace(self.imputer_values_overall, np.NaN)
                imputer_values_overall = np.NaN
            # initialization
            imupter = data_impute(imputer_values_overall, self.imputer_strategy_overall, self.imputer_fillvalue_overall)
            self.train = pd.DataFrame(imupter.fit_transform(self.train), columns=self.columns, index=self.train_ids)
            if self.val is not None:
                self.val = pd.DataFrame(imupter.transform(self.val), columns=self.columns, index=self.val_ids)
            self.model_file.update({"imputer": imupter})
            logger.info("Overall imputation done")
        elif self.feature_flag:
            # if different features have different missing_values
            imputer_list = {}
            pd.Series(self.impute_dict.keys()).apply(lambda x: imputer_series(self.train, x, "train"))
            if self.val is not None:
                pd.Series(self.impute_dict.keys()).apply(lambda x: imputer_series(self.val, x, "val"))
            self.model_file.update({"imputer": imputer_list})
            logger.info("Imputation for features done")

    def onehoter(self):

        def onehot_series(col, flag):
            if flag == "train":
                onehot = OneHotEncoder(handle_unknown='ignore')
                onehot.fit(self.train[[col]])  # first transform the elements to string
                new_data = pd.DataFrame(onehot.transform(self.train[[col]]).toarray())
                onehot_list[col] = onehot
                col_len = len(onehot.categories_[0])
                col_name = ["{}_{}".format(col, i) for i in range(col_len)]
                new_data.columns = col_name
                new_data.index = self.train.index
                self.train = self.train.join(new_data).drop(columns=col)
            elif flag == "val":
                new_data = pd.DataFrame(onehot_list[col].transform(self.val[[col]]).toarray())
                col_name = ["{}_{}".format(col, i) for i in range(len(onehot_list[col].categories_[0]))]
                new_data.columns = col_name
                new_data.index = self.val.index
                self.val = self.val.join(new_data).drop(columns=col)

        if len(self.onehot_feat_conf) > 0:
            onehot_list = {}
            pd.Series(self.onehot_feat_conf.keys()).apply(lambda x: onehot_series(x, "train"))
            if self.val is not None:
                pd.Series(self.onehot_feat_conf.keys()).apply(lambda x: onehot_series(x, "val"))
            self.model_file.update({"onehot": onehot_list})
            logger.info("Onehot for features done")

    def fit(self) -> None:
        """
        missing_values and outlier_values are combined to transform the data
        """
        if len(self.missing_conf) == 0 and len(self.outlier_conf) == 0:
            logger.info("No missing values and outlier values need to be imputed")
        else:
            logger.info("Missing values or outlier values will be imputed")
            self.impute()
            logger.info("Imputation done")
        if len(self.onehot_conf) == 0:
            logger.info("No onehot process")
        else:
            logger.info("Onehot will starts")
            self.onehoter()
            logger.info("Onehot done")
        # recover label column
        if self.label_name is not None:
            self.train = self.train_label.join(self.train)
            if self.val is not None:
                self.val = self.val_label.join(self.val)
        # save model file (optional)
        if len(self.save_model) > 0:
            save_model_config(stage_model_config=self.export_conf,
                              save_path=self.save_dir)
            torch.save(self.model_file, self.save_model_path)
            logger.info("Model file saved")
        # save transformed data
        if len(self.save_trainset_name) > 0:
            path = Path(self.save_trainset_name["path"])
            if not os.path.exists(path):
                path.mkdir(parents=True, exist_ok=True)
            save_train_path = self.save_trainset_name["path"] / Path(self.save_trainset_name["name"])
            self.train.to_csv(save_train_path, index=self.save_trainset_name["has_id"])
            logger.info("Preprocessed trainset done")
        if self.val is not None:
            save_val_path = self.save_valset_name["path"] / Path(self.save_valset_name["name"])
            self.val.to_csv(save_val_path, index=self.save_valset_name["has_id"])
            logger.info("Preprocessed valset done")
