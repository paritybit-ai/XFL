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
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.utils import resample

from algorithm.framework.vertical.sampler.base import VerticalSamplerBase
from common.utils.logger import logger


class VerticalSamplerLabelTrainer(VerticalSamplerBase):
    """
    local feature sampler
    method: str, "random" or "stratify", default: "random";

    strategy: str, "downsample" or "upsample", default: "downsample";

    fraction: int, float or str (sampling ratios of each category),
    e.g. "[(0,0.1), (1,0.2)]",
    default: 0.1;

    random_state: int, RandomState instance or None, default=None
    """

    def __init__(self, train_conf):
        super().__init__(train_conf)
        self.label_count = {}
        self.label_idset = {}
        self._parse_config()
        self._init_data()
        self.sample_ids = []

    def _parse_config(self) -> None:
        self.random_state = self.train_params.get("random_state", None)
        self.method = self.train_params.get("method", "random")
        self.strategy = self.train_params.get("strategy", "downsample")
        self.fraction = self.train_params.get("fraction", 0.1)
        # whether for new customers filter
        self.infer_params = self.train_info.get("infer_params", {})
        if len(self.infer_params) > 0:
            self.threshold_method = self.infer_params["threshold_method"]
            self.threshold = self.infer_params["threshold"]

        if isinstance(self.fraction, str):
            self.fraction = eval(self.fraction)
            if not isinstance(self.fraction, list):
                raise NotImplementedError("Fraction type {} is not supported.".format(self.fraction))

    def fraction_transform(self) -> Any:

        def fraction_num(fraction, num):
            frac_num = int(fraction * num)
            if self.strategy == "downsample":
                if fraction < 0 or fraction > 1:
                    raise ValueError("Fraction should be a numeric number between 0 and 1")
                return max(1, frac_num)
            elif self.strategy == "upsample":
                if fraction < 0:
                    raise ValueError("Fraction should be a numeric number larger than 0")
                return frac_num

        if isinstance(self.fraction, float) or (isinstance(self.fraction, int) and self.strategy == "upsample"):
            tmp = fraction_num(self.fraction, len(self.data))
            return tmp
        elif isinstance(self.fraction, list):
            tmp = [(tup[0], fraction_num(tup[1], self.label_count[tup[0]])) for tup in self.fraction]
            return tmp
        elif isinstance(self.fraction, int) and self.strategy == "downsample":
            return self.fraction

    def random_method(self) -> Any:
        sample_num = self.fraction_transform()
        if self.strategy == "downsample":
            sample_ids = resample(self.data.index,
                                  replace=False,
                                  n_samples=sample_num,
                                  random_state=self.random_state)
            new_data = self.data.loc[sample_ids]
            logger.info("Downsample completed.")
            return sample_ids, new_data
        elif self.strategy == "upsample":
            sample_ids = resample(self.data.index,
                                  replace=True,
                                  n_samples=sample_num,
                                  random_state=self.random_state)
            new_data = self.data.loc[sample_ids]
            logger.info("Upsample completed.")
            return sample_ids, new_data
        else:
            raise NotImplementedError("Strategy type {} is not supported.".format(self.strategy))

    def stratify_method(self) -> Any:
        sample_num = self.fraction_transform()
        sample_ids = []
        new_data = pd.DataFrame()
        if self.strategy == "downsample":
            for label, label_num in sample_num:
                sample_ids_ = resample(self.label_idset[label],
                                       replace=False,
                                       n_samples=label_num,
                                       random_state=self.random_state)
                new_data_ = self.data.loc[sample_ids_]
                sample_ids += sample_ids_
                new_data = pd.concat([new_data, new_data_])
            logger.info("Downsample completed.")
            return sample_ids, new_data
        elif self.strategy == "upsample":
            for label, label_num in sample_num:
                sample_ids_ = resample(self.label_idset[label],
                                       replace=True,
                                       n_samples=label_num,
                                       random_state=self.random_state)
                new_data_ = self.data.loc[sample_ids_]
                sample_ids += sample_ids_
                new_data = pd.concat([new_data, new_data_])
            logger.info("Upsample completed.")
            return sample_ids, new_data
        else:
            raise NotImplementedError("Strategy type {} is not supported.".format(self.strategy))

    def fit(self) -> None:
        new_data = None
        # for most cases
        if len(self.infer_params) == 0:
            if self.method == "random":
                self.sample_ids, new_data = self.random_method()
                logger.info("Random sampler completed.")
            elif self.method == "stratify":
                self.label_count = self.data.groupby(self.label_name)[self.label_name].count().to_dict()
                self.label_idset = self.data.groupby(self.label_name).apply(lambda group: list(group.index)).to_dict()
                self.sample_ids, new_data = self.stratify_method()
                logger.info("Stratify sampler completed.")
            else:
                raise NotImplementedError("Method type {} is not supported.".format(self.method))
        # for new customers filter
        elif len(self.infer_params) > 0:
            key = self.data.columns[0]
            if self.threshold_method == "percentage":
                threshold_num = len(self.data) * self.threshold
                self.sample_ids = self.data.sort_values(by=key, ascending=False).iloc[:int(threshold_num)].index
            elif self.threshold_method == "number":
                if self.threshold > len(self.data):
                    raise OverflowError("Threshold number {} is larger than input data size.".format(self.threshold))
                else:
                    self.sample_ids = self.data.sort_values(by=key, ascending=False).iloc[:int(self.threshold)].index
            elif self.threshold_method == "score":
                self.sample_ids = self.data[self.data[key] > self.threshold].index
            else:
                raise NotImplementedError("Method type {} is not supported.".format(self.threshold_method))

        # save
        if new_data is not None:
            new_data.to_csv(self.save_data_path, index=self.output["trainset"]["has_id"])
            logger.info("Data saved.")
        if len(self.save_id) > 0:
            save_id_path = self.output["model"]["path"] / Path(self.output["model"]["name"])
            with open(save_id_path, "w") as wf:
                json.dump(list(self.sample_ids), wf)
            logger.info("Sample ids saved.")
        # send ids to trainer
        self.broadcast_channel.broadcast(self.sample_ids)
