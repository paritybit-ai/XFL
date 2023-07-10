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
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger
from sklearn.preprocessing import LabelEncoder


def fintech_cut(ser, bins, nan_l, method):
    # all labels begins with 1
    def rebin_freq_2(threshold, num):
        if num <= threshold[0]:
            return 1
        elif threshold[0] < num <= threshold[1]:
            return 2
        else:
            return 3

    def ret_bin(ser_, cut_v):
        l_tmp = [float("%.6f" % i) for i in list(cut_v[1])]
        if len(cut_v[1]) > 2:
            cut_bins = [-np.inf] + l_tmp[1:-1] + [np.inf]
            if method == "equal_frequency":
                cut_ser = pd.DataFrame(LabelEncoder().fit_transform(
                    cut_v[0])).set_index(cut_v[0].index)[0]
                cut_ser = cut_ser + 1
            elif method == "equal_width":
                cut_ser = cut_v[0]
                cut_ser = cut_ser.cat.remove_unused_categories()
        else:
            # never equal_width
            cut_bins = [-np.inf] + [float("%.6f" % i)
                                    for i in l_tmp] + [np.inf]
            cut_ser = ser_.apply(lambda x: rebin_freq_2(cut_v[1], x))
        return cut_ser, cut_bins

    def range_map(nums, _tmp, i):
        if i > 0:
            left = _tmp[nums[i - 1] - 1][1]
            right = _tmp[nums[i] - 1][1]

        else:
            left = _tmp[num[i] - 1][0]
            right = _tmp[num[i] - 1][1]
        return f"({left}, {right}]", [left, right]
    # split nan and norm
    col_dict = {}
    ser_name = ser.name
    bins_ser = []
    ser_cut = ser.loc[ser[~ser.isin(nan_l)].index]
    nan_ind = ser[ser.isin(nan_l)].index
    ser_nan = ser.loc[nan_ind]
    if len(ser_nan) > 0:
        nan_value = float("%.6f" % ser_nan.loc[np.min(nan_ind)])
        bins_ser.append(nan_value)
        ser_nan = pd.DataFrame(np.zeros(len(ser_nan)) +
                               nan_value).set_index(ser_nan.index)[0]
        col_dict[nan_value] = nan_value
    else:
        pass

    if len(ser_cut) > 0:
        # first cut
        cut_value = None
        if len(set(ser_cut)) > 1:
            if method == "equal_width":
                cut_value = pd.cut(ser_cut, bins, retbins=True, labels=[
                                   i + 1 for i in range(bins)])
            elif method == "equal_frequency":
                cut_value = pd.qcut(
                    ser_cut, bins, retbins=True, duplicates='drop')
        elif len(set(ser_cut)) == 1:
            cut_value = (pd.Series(len(ser_cut)*[1]), list(set(ser_cut)))
        cut_ser_, cut_bin = ret_bin(ser_cut, cut_value)
        # label-range map
        tmp_ = [[cut_bin[i], cut_bin[i + 1]] for i in range(len(cut_bin) - 1)]
        num = sorted(set(cut_ser_))
        tt = []
        col_dict.update(dict(zip([num[i] for i in range(len(num))], [
                        range_map(num, tmp_, i)[0] for i in range(len(num))])))
        for i in range(len(num)):
            tt += range_map(num, tmp_, i)[1]
        cut_bin_final = sorted(set(tt))

    # concat nan and cut
        ser = pd.concat([ser_nan, cut_ser_]).loc[ser.index]
        bins_ser = bins_ser + cut_bin_final
        # rename the rightest value to -inf/inf
        if bins_ser[-1] != np.inf:
            bins_ser = bins_ser[:-1] + [np.inf]
            col_dict[list(col_dict.keys(
            ))[-1]] = f"{col_dict[list(col_dict.keys())[-1]].split(' ')[0]} {np.inf}]"
        if bins_ser[0] != -np.inf and bins_ser[0] not in nan_l:
            bins_ser = [-np.inf] + bins_ser[1:]
            col_dict[list(col_dict.keys())[
                0]] = f"({-np.inf}, {col_dict[list(col_dict.keys())[0]].split(' ')[1]}]"
    else:
        ser = ser_nan

    ser.name = ser_name
    return ser, bins_ser, {ser.name: col_dict}


class VerticalBinningWoeIvBase(TrainConfigParser):
    def __init__(self, train_conf: dict, label: bool = False, *args, **kwargs):
        """[summary]

        Args:
            train_conf (dict): [description]
        """
        super().__init__(train_conf)
        self.train_conf = train_conf
        self.df = None
        self.val = None
        self.label = label
        self.woe_map = {}
        self.binning_split = {}
        if self.interaction_params:
            self.save_model = self.interaction_params.get("save_model", False)
        else:
            self.save_model = False
        self.save_dir = Path(self.output.get("path"))
        self.transform_switch = False
        self._init_data()
        if self.save_model:
            self.save_model_name = self.output.get("model").get("name")
            self.export_conf = [{
                "class_name": "VerticalBinningWoeIv",
                "filename": self.save_model_name,
                "bins": self.train_params["binning"]["bins"],
                "input_schema": ','.join([_ for _ in self.df.columns if _ not in set(["y", "id"])]),
            }]
        self.feature_binning()

    def _init_data(self) -> None:
        """Load data: input data with id

        Returns:

        """
        logger.info("Start reading data.")
        if self.input_trainset[0].get("has_id", True):
            # index_col = self.input_trainset[0].get("index_col", 'id')
            index_col = 0
        else:
            index_col = None

        if self.input_trainset[0]["type"] == "csv":
            file_path = str(
                Path(self.input_trainset[0]["path"], self.input_trainset[0]["name"]))
            if self.computing_engine == "local":
                self.df = pd.read_csv(file_path, index_col=index_col)

        logger.info("Reading data successfully.")

        if self.input_trainset[0].get("has_label", False):
            # self.y = self.df[label_name]
            # self.df = self.df.drop(label_name, axis=1)
            self.df.columns = ["y"] + list(self.df.columns[1:])
            self.y = self.df.iloc[:, 0]
            self.df = self.df.iloc[:, 1:]
        else:
            self.y = None

        # read val for transform
        if len(self.input_valset) > 0:
            self.transform_switch = True
            self.val = pd.read_csv(
                str(Path(self.input_valset[0]["path"], self.input_valset[0]["name"])), index_col=index_col)

    def feature_binning(self) -> None:
        """Parse and execute feature binning.

        Returns: None

        """
        logger.info("Start binning")
        nan_l = self.input_trainset[0]["nan_list"]
        method = self.train_params["binning"]['method']
        bin_num = self.train_params["binning"]["bins"]
        time_start = time.time()
        tmp = pd.Series(self.df.columns).apply(
            lambda x: fintech_cut(self.df[x], bin_num, nan_l, method))

        def get_cut_result(result):
            self.df[result[0].name] = result[0]
            self.binning_split.update({result[0].name: result[1]})
            self.woe_map.update(result[2])

        tmp.apply(lambda x: get_cut_result(x))
        time_end = time.time()
        logger.info("Cost time of binning is: {}s".format(
            time_end - time_start))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        file_path = f'{self.save_dir}/{self.output["split_points"]["name"]}'
        with open(file_path, "w") as wf:
            json.dump(self.binning_split, wf)
        logger.info("Binning split points saved as {}.".format(file_path))

        if isinstance(self.y, pd.Series):
            self.df = pd.concat([self.y, self.df], axis=1)
