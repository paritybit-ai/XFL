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

from common.utils.data_utils import download_url, pd_train_test_split


class WDBC(torch.utils.data.Dataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    md5 = None
    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    datapath = os.path.join(dirpath, "wdbc.data")
    datadir = "breast_cancer_wisconsin"

    def __init__(
        self,
        redownload: bool = False,
    ) -> None:

        super().__init__()
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self._download(redownload)
        self.data = pd.read_csv(self.datapath, names=[
                                "id", "y"] + [f"x{i:0>2d}" for i in range(30)])
        self.data["y"] = self.data["y"].map({"M": 1, "B": 0})
        self.data[[f"x{i:0>2d}" for i in range(30)]] = self.data[[
            f"x{i:0>2d}" for i in range(30)]].apply(lambda x: (x-x.mean())/x.std())

    def __getitem__(self, index: int) -> Any:
        return self.data[[f"x{i:0>2d}" for i in range(30)]].values[index], self.data["y"].values[index]

    def __len__(self) -> int:
        return len(self.data.values)

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
        drop_ratio = reallocate_dict['drop_ratio']
        random_state = reallocate_dict["random_seed"]
        parties = reallocate_dict["parties"]
        np.random.seed(random_state)

        final_dir_path = os.path.join(
            self.dirpath, self.datadir, reallocate_folder)
        if not os.path.exists(final_dir_path):
            os.makedirs(final_dir_path)
        if mode == "vertical":
            cols = [f"x{i:0>2d}" for i in range(30)]
            split_cols = np.array_split(cols, splits)
            for i, span in enumerate(split_cols):
                if "labeled" in parties[i]:
                    train_data, test_data = pd_train_test_split(
                        self.data[["id", "y"]+list(span)], test_ratio=test_ratio, random_state=random_state)
                else:
                    train_data, test_data = pd_train_test_split(
                        self.data[["id"]+list(span)], test_ratio=test_ratio, random_state=random_state)

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
            # rand_idx = np.random.permutation(range(len(train_data)))
            # indices = np.array_split(rand_idx, splits)
            indices = np.array_split(range(len(train_data)), splits)
            for i, party in enumerate(parties):
                csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{party}.csv')
                data = train_data.loc[indices[i]]
                data.to_csv(csv_path, index=False)

        elif mode == "transfer":
            all_train_data, all_test_data = pd_train_test_split(
                self.data, test_ratio=test_ratio, random_state=random_state)
            cols = [f"x{i:0>2d}" for i in range(30)]
            split_cols = np.array_split(cols, splits)
            index_dict = {}
            
            for i, span in enumerate(split_cols):
                if parties[i] == "labeled":
                    train_data, _ = pd_train_test_split(
                        all_train_data[["id", "y"]+list(span)], test_ratio=drop_ratio, 
                        shuffle=True, random_state=random_state+100
                    )
                    test_data = all_test_data[["id", "y"]+list(span)]
                else:
                    train_data, _ = pd_train_test_split(
                        all_train_data[["id"]+list(span)], test_ratio=drop_ratio, 
                        shuffle=True, random_state=random_state+200
                    )
                    test_data = all_test_data[["id"]+list(span)]
                index_dict[i] = train_data["id"].to_list()

                train_csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{parties[i]}_train.csv')
                test_csv_path = os.path.join(
                    final_dir_path, f'{self.datadir}_{parties[i]}_test.csv')
                train_data.to_csv(train_csv_path, index=False)
                test_data.to_csv(test_csv_path, index=False)

            overlap_index = index_dict[0]
            for i in range(1, len(index_dict)):
                overlap_index = self.intersection_list(overlap_index, index_dict[i])
            np.save(os.path.join(final_dir_path, "overlap_index.npy"), np.array(overlap_index))

        os.remove(self.datapath)
    
    @staticmethod
    def intersection_list(list1, list2):
        list3 = [value for value in list1 if value in list2]
        return list3


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="vertical",
                        help="vertical, horizontal or transfer task")
    parser.add_argument("--splits", type=int, default=2,
                        help="number of parties")
    parser.add_argument("--test_ratio", type=float,
                        default=0.3, help="ratio of test data")
    parser.add_argument("--drop_ratio", type=float,
                        default=0.3, help="ratio of drop data")
    parser.add_argument("--random_seed", type=int,
                        default=0, help="random seed")
    parser.add_argument("--party", nargs="+", help="involved parties")
    args = parser.parse_args()

    reallocate_dict = {
        "mode": args.mode,
        "splits": args.splits,
        "test_ratio": args.test_ratio,
        "drop_ratio": args.drop_ratio,
        "random_seed": args.random_seed,
        "parties": args.party
    }

    wdbc = WDBC()
    wdbc.reallocate(reallocate_dict)
