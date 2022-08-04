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

from algorithm.core.data_io import CsvReader
from common.communication.gRPC.python.channel import BroadcastChannel
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger


class VerticalSamplerBase(TrainConfigParser):
    def __init__(self, train_conf: dict):
        super().__init__(train_conf)
        self._init_data()
        self.broadcast_channel = BroadcastChannel(name="vertical_sampler_channel")
        self.save_id = self.output.get("model", {})
        self.save_data_path = self.output["trainset"]["path"] / Path(self.output["trainset"]["name"])
        if not os.path.exists(self.output["trainset"]["path"]):
            os.makedirs(self.output["trainset"]["path"])

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
        if len(self.input["dataset"]) > 0:
            data: CsvReader = self.__load_data(self.input["dataset"])
            self.data = data.table.set_index(data.ids)
            self.label_name = data.label_name()
        else:
            raise NotImplementedError("Dataset was not configured.")
