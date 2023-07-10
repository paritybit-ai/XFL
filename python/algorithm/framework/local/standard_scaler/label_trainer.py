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
from google.protobuf import json_format
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from common.utils.utils import save_model_config
from common.model.python.feature_model_pb2 import StandardizationModel
from service.fed_control import ProgressCalculator
from common.utils.model_io import ModelIO


class LocalStandardScalerLabelTrainer(TrainConfigParser):
    """
    local standard scaler
    removing the mean and scaling to unit variance
            z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation of the training samples.
    """

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
            "class_name": "LocalStandardScaler",
            "filename": self.save_pmodel_name,
            "input_schema": ','.join([_ for _ in self.train_data.columns if _ not in set(["y", "id"])]),
        }]

    def _parse_config(self) -> None:
        self.save_dir = self.output.get("path")
        self.save_model_name = self.output.get("model", {}).get("name")
        self.save_pmodel_name = self.output.get(
            "proto_model", {}).get("name", "")
        self.save_trainset_name = self.output.get("trainset", {}).get("name")
        self.save_valset_name = self.output.get("valset", {}).get("name")

    def _init_data(self) -> None:
        """
        load raw data
        1. using train set to generate the standard scaler
        2. apply it to the valid set for subsequent model training
        :return:
        """
        if self.input_trainset:
            df_list = []
            for ts in self.input_trainset:
                if ts.get("type") == "csv":
                    df_list.append(pd.read_csv(
                        os.path.join(ts.get("path"), ts.get("name"))))
                    if ts.get("has_id") and 'id' not in self.skip_cols:
                        self.skip_cols.append('id')
                    if ts.get("has_label") and 'y' not in self.skip_cols:
                        self.skip_cols.append('y')
                else:
                    raise NotImplementedError(
                        "Load function {} does not Implemented.".format(
                            ts.get("type"))
                    )
            self.train_data = pd.concat(df_list)
            self.skip_cols.extend(
                self.train_data.columns[self.train_data.dtypes == 'object'])
            if len(self.skip_cols) > 0:
                logger.info("Skip columns {}".format(','.join(self.skip_cols)))

        if self.input_valset:
            df_list = []
            for vs in self.input_valset:
                if vs.get("type") == "csv":
                    df_list.append(pd.read_csv(
                        os.path.join(vs.get("path"), vs.get("name"))))
                    self.transform_switch = True
            self.valid_data = pd.concat(df_list)

    def fit(self) -> None:
        """
        train a standard scaler

        params:
        - with_mean -> Boolean, u = 0 if with_mean=False
        - with_std -> Boolean, s = 1 if with_std=False

        output:
        - the .csv files which save the transformed data
        - the .pt file which saves the normalizer

        :return: None
        """
        if self.train_data is None:
            logger.info("no data, skip stage.".format(self.identity))
            return
        scaler_dict = {}
        cols = [_ for _ in self.train_data.columns if _ not in self.skip_cols]
        standardize_dict = {}
        standard_scaler = {}
        default_with_mean = self.train_params.get("with_mean")
        default_with_std = self.train_params.get("with_std")
        if default_with_mean is None:
            logger.warning(
                "cannot find the param with_mean, skip global standardization.")
        elif default_with_std is None:
            logger.warning(
                "cannot find the param with_std, skip global standardization.")
        else:
            for f in cols:
                standardize_dict[f] = {
                    "with_mean": default_with_mean,
                    "with_std": default_with_std
                }
        for f in self.train_params.get("feature_standard", []):
            if self.train_params["feature_standard"][f].get("with_mean") is None:
                logger.warning(
                    "cannot find the param with_mean for feature {}, skip standardization.".format(f))
            elif self.train_params["feature_standard"][f].get("with_std") is None:
                logger.warning(
                    "cannot find the param with_std for feature {}, skip standardization.".format(f))
            elif f not in cols:
                raise KeyError("Feature {} cannot be found in df.".format(f))
            else:
                standardize_dict[f] = standardize_dict.get(f, {})
                standardize_dict[f]["with_mean"] = self.train_params["feature_standard"][f]["with_mean"]
                standardize_dict[f]["with_std"] = self.train_params["feature_standard"][f]["with_std"]
        for idx, (f, d) in enumerate(standardize_dict.items()):
            logger.info("{}: Count={}, Min={}, Max={}, Unique={}.".format(
                f, self.train_data[f].count(), self.train_data[f].min(),
                self.train_data[f].max(), self.train_data[f].nunique()
            ))
            if d["with_mean"]:
                u = float(self.train_data[f].mean())
            else:
                u = 0
            if d["with_std"]:
                s = float(self.train_data[f].std())
            else:
                s = 1
            if np.abs(s - 0) < 1e-6:
                s = 1
            self.train_data[f] = (self.train_data[f] - u) / s
            if self.transform_switch:
                self.valid_data[f] = (self.valid_data[f] - u) / s
            logger.info("{}: u={}, s={}.".format(f, u, s))
            standard_scaler[idx] = {"feature": f, "u": u, "s": s}
        scaler_dict["standard_scaler"] = standard_scaler
        self.save(scaler_dict)
        ProgressCalculator.finish_progress()

    def save(self, scaler_dict):
        if self.save_dir:
            self.save_dir = Path(self.save_dir)
        else:
            return

        if self.save_pmodel_name:
            save_model_config(stage_model_config=self.export_conf,
                              save_path=self.save_dir)
            dump_path = self.save_dir / Path(self.save_pmodel_name)
            standard_model = StandardizationModel()
            json_format.ParseDict(scaler_dict, standard_model)
            
            with open(dump_path, 'wb') as f:
                f.write(standard_model.SerializeToString())
                
            logger.info("Standardize results saved as {}.".format(dump_path))
                
        if self.save_model_name:
            ModelIO.save_json_model(
                model_dict=scaler_dict,
                save_dir=self.save_dir,
                model_name=self.save_model_name)
            
            logger.info(
                "Standardize results saved as {}.".format(Path(self.save_dir) / self.save_model_name))

        save_trainset_path = self.save_dir / Path(self.save_trainset_name)
        if not os.path.exists(os.path.dirname(save_trainset_path)):
            os.makedirs(os.path.dirname(save_trainset_path))
        self.train_data.to_csv(
            save_trainset_path, float_format='%.6g', index=False)
        logger.info("Data saved as {}, length: {}.".format(
            save_trainset_path, len(self.train_data)))

        if self.transform_switch:
            save_valset_path = self.save_dir / Path(self.save_valset_name)
            if not os.path.exists(os.path.dirname(save_valset_path)):
                os.makedirs(os.path.dirname(save_valset_path))
            self.valid_data.to_csv(
                save_valset_path, float_format='%.6g', index=False)
            logger.info("Data saved as {}, length: {}.".format(
                save_valset_path, len(self.valid_data)))

        logger.info("Data standardize completed.")
