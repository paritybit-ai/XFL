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

from algorithm.framework.vertical.sampler.base import VerticalSamplerBase
from common.utils.logger import logger


class VerticalSamplerTrainer(VerticalSamplerBase):
    def __init__(self, train_conf):
        super().__init__(train_conf)

    def fit(self) -> None:
        # receive sample_ids
        sample_ids = self.broadcast_channel.recv()
        new_data = self.data.loc[sample_ids]
        if len(self.save_id) > 0:
            save_id_path = self.output["path"] / Path(self.output["sample_id"]["name"])
            if not os.path.exists(os.path.dirname(save_id_path)):
                os.makedirs(os.path.dirname(save_id_path))
            with open(save_id_path, "w") as wf:
                json.dump(list(sample_ids), wf)
            logger.info("Sample ids saved to {}.".format(save_id_path))

        new_data.to_csv(self.save_data_path, index=self.input["dataset"][0]["has_id"])
        logger.info("Data saved to {}.".format(self.save_data_path))
