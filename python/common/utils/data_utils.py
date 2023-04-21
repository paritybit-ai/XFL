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


import gzip
import hashlib
import os
import pathlib
import ssl
import tarfile
import zipfile
from typing import Optional
from urllib import request

from sklearn.utils import shuffle as sk_shuffle


def cal_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    elif md5 is None:
        return True
    else:
        return cal_md5(fpath) == md5


def download_url(url: str, fpath: str, md5: str, chunk_size: int = 1024 * 32) -> None:
    if check_integrity(fpath, md5):
        print("Verified dataset Already exists")
        return
    print("Dataset downloading...")
        
    with request.urlopen(request.Request(url), context=ssl._create_unverified_context()) as response:
        with open(fpath, "wb") as fh:
            for chunk in iter(lambda: response.read(chunk_size), b""):
                if not chunk:
                    continue
                fh.write(chunk)
        fh.close()


def extract_file_recursively(from_path: str, to_path: str) -> None:
    def extract(from_path, to_path, suffix):
        if suffix == ".tar":
            with tarfile.open(from_path,  "r") as tar:
                tar.extractall(to_path)
        elif suffix == ".gz":
            with gzip.open(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
                wfh.write(rfh.read())

    suffixes = pathlib.Path(from_path).suffixes
    suffix = suffixes[-1]

    if len(suffixes) == 1:
        if suffix not in [".gz",".tar"]:
            return 
        extract(from_path, to_path, suffix)
        os.remove(from_path)
        return
    else:
        _to_path = pathlib.Path(from_path).parent.joinpath(pathlib.Path(from_path).stem)
        extract(from_path, _to_path, suffix)
        os.remove(from_path)
        from_path = _to_path
    extract_file_recursively(from_path, to_path)


def download_and_extract_data(url: str, md5: str, data_path: str, data_folder: Optional[str] = None, to_path: Optional[str] = None) -> None:
    if not to_path:
        to_path = pathlib.Path(data_path).parent
    if data_folder:
        final_path = os.path.join(to_path, data_folder)
        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            print("Dataset has already existed")
            return
    download_url(url, data_path, md5)
    extract_file_recursively(data_path, to_path)
    print("Data finished downloading and extraction")


def pd_train_test_split(df, test_ratio: float, shuffle: bool = False, random_state: int = None):
    if shuffle:
        df = sk_shuffle(df, random_state=random_state)
    train_df = df[int(len(df)*test_ratio):].reset_index(drop=True)
    test_df = df[:int(len(df)*test_ratio)].reset_index(drop=True)
    return train_df, test_df

