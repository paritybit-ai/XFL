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
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.utils.data_utils import download_and_extract_data, pd_train_test_split


class SST2(torch.utils.data.Dataset):
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    md5 = "9f81648d4199384278b86e315dac217c"
    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    datapath = os.path.join(dirpath, "SST-2.zip")
    data_folder = "SST-2"
    raw_data_folder = os.path.join(dirpath, data_folder, "original")

    def __init__(
        self,
        redownload: bool = False,
        mode: str = "train"
    ) -> None:

        super().__init__()
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self._download_and_extract(redownload)
        self.mode = mode 
        self.train = pd.read_csv(os.path.join(self.dirpath, self.data_folder, "train.tsv"),sep='\t')
        self.val = pd.read_csv(os.path.join(self.dirpath, self.data_folder, "dev.tsv"), sep='\t')
        self.test = pd.read_csv(os.path.join(self.dirpath, self.data_folder, "test.tsv"), sep='\t')
        self.data = getattr(self, mode)
 
    def __getitem__(self, index: int) -> Any:
        return self.data["sentence"].values[index], self.data["label"].values[index]

    def __len__(self) -> int:
        return len(self.data["sentence"].values)
    
    def get_data(self):
        return self.data["sentence"].values, self.data["label"].values

    def _download_and_extract(self, redownload):
        if redownload:
            shutil.rmtree(os.path.join(self.dirpath, self.data_folder))
        download_and_extract_data(
            self.url, self.md5, self.datapath, data_folder=self.data_folder)

    def reallocate(self, reallocate_dict):
        mode = reallocate_dict['mode']
        splits = reallocate_dict['splits']
        reallocate_folder = f'{splits}party'
        random_state = reallocate_dict["random_seed"]
        parties = reallocate_dict["parties"]
        np.random.seed(random_state)

        final_dir_path = os.path.join(
            self.dirpath, self.data_folder, reallocate_folder)
        if not os.path.exists(final_dir_path):
            os.makedirs(final_dir_path)
        if mode == "vertical":
            raise NotImplementedError("SST-2 task currently doesn't support vertical federated learning")

        elif mode == "horizontal":
            val_path = os.path.join(
                final_dir_path, f'{self.data_folder}_val.tsv')
            self.val.to_csv(val_path, index=False, sep="\t")
            rand_idx = np.random.permutation(range(len(self.train)))
            indices = np.array_split(rand_idx, splits)
            for i, party in enumerate(parties):
                tsv_path = os.path.join(
                    final_dir_path, f'{self.data_folder}_{party}.tsv')
                data = self.train.loc[indices[i]]
                data.to_csv(tsv_path, index=False, sep='\t')
        shutil.rmtree(self.raw_data_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="horizontal",
                        help="vertical or horizontal task")
    parser.add_argument("--splits", type=int, default=2,
                        help="number of parties")
    parser.add_argument("--random_seed", type=int,
                        default=0, help="random seed")
    parser.add_argument("--party", nargs="+", help="involved parties")
    args = parser.parse_args()

    reallocate_dict = {
        "mode": args.mode,
        "splits": args.splits,
        "random_seed": args.random_seed,
        "parties": args.party
    }

    sst2= SST2()
    sst2.reallocate(reallocate_dict)
