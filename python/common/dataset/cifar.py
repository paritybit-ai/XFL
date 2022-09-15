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


import argparse
import os
import pickle
import shutil
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from common.utils.data_utils import check_integrity, download_and_extract_data
from PIL import Image
import torchvision.transforms as transforms



class CIFAR10(torch.utils.data.Dataset):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    md5 = "c58f30108f718f92721af3b95e74349a"
    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    datapath = os.path.join(dirpath, "cifar-10-python.tar.gz")
    data_folder = "cifar-10-batches-py"
    data_folder_renamed = "cifar10"
    train_dict = {
        "data_batch_1": "c99cafc152244af753f735de768cd75f",
        "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
        "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
        "data_batch_4": "634d18415352ddfa80567beed471001a",
        "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    }

    test_dict = {
        "test_batch": "40351d587109b95175f43aff81a1287e",
    }

    metadata = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        train: bool = True,
        redownload: bool = False,
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__()
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self.train = train
        self._download_and_extract(redownload)
        self._load_metadata()
        self.data_dict = self.train_dict if self.train else self.test_dict
        self.transform = transform
        self.data = []
        self.labels = []
        for file_name, md5 in self.data_dict.items():
            file_path = os.path.join(self.dirpath, self.data_folder, file_name)
            if not check_integrity(file_path, md5):
                self.intergrity = False
                raise RuntimeError(
                    f'{file_path} has been corruptted or lost. Please redownload the data by setting redownload=True')
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.labels.extend(entry["labels"])
                else:
                    self.labels.extend(entry["fine_labels"])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))  # HWC format
        self.labels = np.array(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, label = self.data[index], self.labels[index]
        data = Image.fromarray(data)
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self) -> int:
        return len(self.data)

    def _download_and_extract(self, redownload):
        if redownload:
            shutil.rmtree(os.path.join(self.dirpath, self.data_folder))

        download_and_extract_data(
            self.url, self.md5, self.datapath, data_folder=self.data_folder)

    def _load_metadata(self) -> None:
        metapath = os.path.join(
            self.dirpath, self.data_folder, self.metadata["filename"])
        if not check_integrity(metapath, self.metadata["md5"]):
            raise RuntimeError(
                "Dataset metadata has been found or corrupted. Please redownload the data by setting redownload=True")
        with open(metapath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.classes = data[self.metadata["key"]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def reallocate(self, reallocate_dict):
        splits = reallocate_dict['splits']
        reallocate_folder = f'{splits}party'
        if not self.train:
            splits = len(reallocate_dict["party"])
        if reallocate_dict["sampling"] == "random":
            np.random.seed(reallocate_dict["seed"])
            if isinstance(reallocate_dict["splits"], int):
                rand_idx = np.random.permutation(range(len(self.data)))
                # split into equal arrays
                indices = np.array_split(rand_idx, reallocate_dict["splits"])
            elif isinstance(reallocate_dict["splits"], list):
                assert sum(
                    reallocate_dict["splits"]) == 1, "the sum of fractions must be 1"
                rand_idx = np.random.permutation(range(len(self.data)))
                sections = np.floor(
                    reallocate_dict["splits"] * len(self.data))
                sections = np.cumsum(sections)[:-1]
                # split into arrays according to ratios
                indices = np.split(rand_idx, sections)

            final_dir_path = os.path.join(
                self.dirpath, self.data_folder_renamed, reallocate_folder)
            if not os.path.exists(final_dir_path):
                os.makedirs(final_dir_path)
            for i, party in enumerate(reallocate_dict["party"]):
                npy_path = os.path.join(
                    final_dir_path, f"{self.data_folder_renamed}_{party}.npz")
                data = self.data[indices[i]]
                labels = self.labels[indices[i]]
                np.savez(npy_path, data=data, labels=labels)
        elif reallocate_dict["sampling"] == "biased":
            np.random.seed(reallocate_dict["seed"])
            indices_group = [[] for _ in range(reallocate_dict["splits"])]
            for group_label, fractions in reallocate_dict["group_fractions"].items():
                group_index = np.where(self.label == group_label)
                group_index = np.random.permutation(group_index)
                sections = np.floor(fractions * len(group_index))
                sections = np.cumsum(sections)[:-1]
                indices = np.split(rand_idx, sections)
                for i, indice in enumerate(indices):
                    indices_group[i].extend(indice)

            final_dir_path = os.path.join(
                self.dirpath, self.data_folder_renamed, reallocate_folder)
            for i, party in enumerate(reallocate_dict["party"]):
                npy_path = os.path.join(final_dir_path, f"{party}.npz")
                if not os.path.exists(final_dir_path):
                    os.makedirs(final_dir_path)
                data = self.data[indices_group[i]]
                labels = self.labels[indices_group[i]]
                np.savez(npy_path, data=data, labels=labels)


class CIFAR100(CIFAR10):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    datapath = os.path.join(dirpath, "cifar-100-python.tar.gz")
    data_folder = "cifar-100-python"
    data_folder_renamed = "cifar-100"
    train_dict = {
        "train": "16019d7e3df5f24257cddd939b257f8d",
    }

    test_dict = {
        "test", "f0ef6b0ae62326f3e7ffdfab6717acfc",
    }
    metadata = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=2,
                        help="number of parties")
    parser.add_argument("--sampling", type=str, default="random",
                        help="mode to split the dataset, random or biased")
    parser.add_argument("--party", nargs="+", help="involved parties")
    parser.add_argument("--keep_raw_data",
                        action='store_true', help="keep raw data file")
    args = parser.parse_args()
    if args.sampling == "random":
        train_reallocate_dict = {
            "sampling": "random",
            "splits": args.splits,
            "seed": 0,
            "party": args.party
        }

        test_reallocate_dict = {
            "sampling": "random",
            "splits": args.splits,
            "seed": 0,
            "party": ["test"]
        }
    elif args.sampling == "biased":
        train_reallocate_dict = {
            "sampling": "biased",
            "splits": args.splits,
            "seed": 0,
            "group_fractions": {1: [0.8, 0.2], 2: [0.8, 0.2], 3: [0.8, 0.2], 4: [0.8, 0.2], 5: [0.8, 0.2], 6: [0.2, 0.8], 7: [0.2, 0.8], 8: [0.2, 0.8], 9: [0.2, 0.8], 0: [0.2, 0.8]},
            "party": args.party
        }

        test_reallocate_dict = {
            "sampling": "random",
            "splits": args.splits,
            "seed": 0,
            "party": ["test"]
        }

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
     

    cifar10_train = CIFAR10(train=True)
    cifar10_train.reallocate(train_reallocate_dict)
    cifar10_test = CIFAR10(train=False)
    cifar10_test.reallocate(test_reallocate_dict)
    if not args.keep_raw_data:
        shutil.rmtree(os.path.join(
            cifar10_train.dirpath, cifar10_train.data_folder))
