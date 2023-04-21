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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
import torch

from common.utils.data_utils import download_url, pd_train_test_split


class Azpro(torch.utils.data.Dataset):
    url = "https://r-data.pmagunia.com/system/files/datasets/dataset-15359.csv"
    md5 = None
    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    datapath = os.path.join(dirpath, "azpro")
    datadir = "azpro_data"

    def __init__(
            self,
            redownload: bool = False,
    ) -> None:

        super().__init__()
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self._download(redownload)
        # raw_df = pd.read_csv("/opt/dataset/data.csv", index_col=0)
        raw_df = pd.read_csv(self.datapath, index_col=None)
        raw_df = raw_df.drop(columns=["hospital"])
        self.feature = raw_df.iloc[:, 1:]
        self.feature_cols = self.feature.columns
        self.label = pd.DataFrame(raw_df.iloc[:, 0])
        self.label.columns = ["y"]
        self.id = np.arange(len(self.label))
        self.data = self.label.join(self.feature)
        self.data = self.data.reset_index()
        self.data.columns = ["id"] + list(self.data.columns[1:])

        if reallocate_dict["norm"]:
            feature = self.data.iloc[:, 1:]
            scaler = MinMaxScaler()
            data_norm = pd.DataFrame(scaler.fit_transform(feature), columns=feature.columns)
            self.data = self.data.iloc[:, :1].join(data_norm)

    def __getitem__(self, index: int) -> Any:
        return self.feature[index], self.label[index]

    def _download(self, redownload):
        if redownload:
            shutil.rmtree(os.path.join(self.dirpath, self.datadir))
        download_url(
            self.url, self.datapath, self.md5)

    def reallocate(self, reallocate_dict):
        mode = reallocate_dict['mode']
        self.datadir = f"{self.datadir}_{mode}"
        splits = reallocate_dict['splits']
        reallocate_folder = f'{splits}party'
        test_ratio = reallocate_dict['test_ratio']
        random_state = reallocate_dict["random_seed"]
        parties = reallocate_dict["parties"]
        np.random.seed(random_state)

        final_dir_path = os.path.join(
            self.dirpath, self.datadir, reallocate_folder)
        if not os.path.exists(final_dir_path):
            os.makedirs(final_dir_path)
        if mode == "vertical":
            cols = self.feature_cols
            split_cols = np.array_split(cols, splits)
            for i, span in enumerate(split_cols):
                if "labeled" in parties[i]:
                    train_data, test_data = pd_train_test_split(
                        self.data[["id", "y"] + list(span)], test_ratio=test_ratio, random_state=random_state)
                else:
                    train_data, test_data = pd_train_test_split(
                        self.data[["id"] + list(span)], test_ratio=test_ratio, random_state=random_state)

                train_csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{parties[i]}_train.csv')
                test_csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{parties[i]}_test.csv')
                train_data.to_csv(train_csv_path, index=False)
                test_data.to_csv(test_csv_path, index=False)

        elif mode == "horizontal":
            train_data, test_data = pd_train_test_split(
                self.data, test_ratio=test_ratio, random_state=random_state)
            test_csv_path = os.path.join(
                final_dir_path, f'{self.datadir}_test.csv')
            test_data.to_csv(test_csv_path, index=False)
            rand_idx = np.random.permutation(range(len(train_data)))
            indices = np.array_split(rand_idx, splits)
            for i, party in enumerate(parties):
                csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{party}.csv')
                data = train_data.loc[indices[i]]
                data.to_csv(csv_path, index=False)
        os.remove(self.datapath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="vertical",
                        help="vertical or horizontal task")
    parser.add_argument("--splits", type=int, default=2,
                        help="number of parties")
    parser.add_argument("--test_ratio", type=float,
                        default=0.3, help="ratio of test data")
    parser.add_argument("--random_seed", type=int,
                        default=0, help="random seed")
    parser.add_argument("--party", nargs="+", help="involved parties")
    parser.add_argument("--norm", type=bool,
                        default=False, help="normalization")
    args = parser.parse_args()

    reallocate_dict = {
        "mode": args.mode,
        "splits": args.splits,
        "test_ratio": args.test_ratio,
        "random_seed": args.random_seed,
        "parties": args.party,
        "norm": args.norm
    }

    boston = Azpro()
    boston.reallocate(reallocate_dict)
