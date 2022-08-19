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
from pathlib import Path

import numpy as np

from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger

import glob
import time
from tqdm import tqdm


def parallel_apply_generator(func, iterable, workers, max_queue_size, dummy=False, random_seeds=True):
    if dummy:
        from multiprocessing.dummy import Pool, Queue, Manager
    else:
        from multiprocessing import Pool, Queue, Manager

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Manager().Queue(), Manager().Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            ii, dd = in_queue.get()
            r = func(dd)
            out_queue.put((ii, r))

    pool = Pool(workers, worker_step, (in_queue, out_queue))

    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            in_queue.put((i, d), block=False)
            break
        if out_queue.qsize() > 0:
            yield out_queue.get()
            out_count += 1
    while out_count != in_count:
        yield out_queue.get()
        out_count += 1
    pool.terminate()


def parallel_apply(
        func,
        iterable,
        workers,
        max_queue_size,
        callback=None,
        dummy=False,
        random_seeds=True,
        unordered=True
):
    generator = parallel_apply_generator(func, iterable, workers, max_queue_size, dummy, random_seeds)

    if callback is None:
        if unordered:
            return [d for i, d in generator]
        # else:
        #     results = sorted(generator, key=lambda d: d[0])
        #     return [d for i, d in results]
    else:
        for i, d in generator:
            callback(d)


class LocalDataSplitLabelTrainer(TrainConfigParser):
    def __init__(self, train_conf):
        """
        support data split from more than one file:
        if there is no input dataset name, all csv files under the input dataset path will be combined.
        Args:
            train_conf:
            shuffle: bool, whether need to shuffle;
            worker_num: int, parallel worker num;
            batch_size: int, the size of small file;
        """
        super().__init__(train_conf)
        # input config
        self.line_num = 0
        self.input_data = self.input.get("dataset", [])
        if self.input_data:
            self.input_data_path = self.input_data[0].get("path", None)
            self.input_data_name = self.input_data[0].get("name", None)
            self.header = self.input_data[0].get("header", False)
        if self.input_data:
            if self.input_data_name is None:
                # more than one file
                self.files = glob.glob("{}/*.csv".format(self.input_data_path))
            else:
                # one file
                self.files = glob.glob("{}/{}".format(self.input_data_path, self.input_data_name))
        else:
            raise NotImplementedError("Dataset was not configured.")
        # output config
        self.save_trainset = self.output.get("trainset", {})
        save_trainset_path = self.save_trainset.get("path", self.input_data_path)
        save_trainset_name = self.save_trainset.get("name", "{}_train.csv".format(
            self.files[0].split("/")[-1].replace(".csv", '')))
        if not os.path.exists(Path(save_trainset_path)):
            Path(save_trainset_path).mkdir(parents=True, exist_ok=True)
        self.save_trainset_name = Path(save_trainset_path, save_trainset_name)
        self.save_valset = self.output.get("valset", {})
        save_valset_path = self.save_valset.get("path", self.input_data_path)
        save_valset_name = self.save_valset.get("name", "{}.csv".format(self.files[0].split("/")[-1].
                                                                        replace(".csv", '')))
        self.save_valset_name = Path(save_valset_path, save_valset_name)
        # train info
        self.shuffle = self.train_params.get("shuffle_params", False)
        if self.shuffle:
            logger.info("Shuffle is needed")
        else:
            logger.info("Shuffle is not needed")
        self.worker_num = self.train_params.get("worker_num", 4)
        if self.shuffle:
            self.batch_size = self.train_params.get("batch_size", 100000)
        self.train_weight = self.train_params.get("train_weight", 8)
        self.val_weight = self.train_params.get("val_weight", 2)
        self.train_ratio = self.train_weight / (self.train_weight + self.val_weight)
        self.header_data = None

    def local_shuffle(self, batch_k):
        batch, k = batch_k
        np.random.shuffle(batch)
        with open("%s_local_shuffle/%05d.csv" % (self.input_data_path, k), "w") as f:
            for text in batch:
                f.write(text)

    def generator(self):
        batch, k = [], 0
        for j in tqdm(self.files, ncols=0, desc="Local Shuffling"):
            header = self.header
            with open(j) as f:
                for line in f:
                    if header:
                        self.header_data = line
                        header = False
                        continue
                    batch.append(line)
                    self.line_num += 1
                    if len(batch) == self.batch_size:
                        yield batch, k
                        batch = []
                        k += 1
        if batch:
            yield batch, k

    def fit(self):
        start_time = time.time()
        if self.shuffle:
            # mkdir of local_shuffle
            temp_path = Path("%s_local_shuffle" % self.input_data_path)
            if not os.path.exists(temp_path):
                temp_path.mkdir(parents=True, exist_ok=True)
            # local shuffle
            parallel_apply(func=self.local_shuffle, iterable=self.generator(), workers=self.worker_num,
                           max_queue_size=10)
            # train and val line num
            trainset_num = int(self.line_num * self.train_ratio)
            files = glob.glob("{}/*.csv".format(temp_path))
            opens = [open(j) for j in files]
            # global shuffle
            n, k = 0, 0
            F = open(self.save_trainset_name, "w")
            if self.header:
                F.write(self.header_data)
            for i in tqdm(range(self.batch_size), ncols=0, desc="Global Shuffling"):
                orders = np.random.permutation(len(opens))
                for j in orders:
                    text = opens[j].readline()
                    if text:
                        n += 1
                        F.write(text)
                        if n == trainset_num:
                            n = 0
                            k += 1
                            F = open(self.save_valset_name, "w")
                            if self.header:
                                F.write(self.header_data)
            shutil.rmtree(temp_path, ignore_errors=True)  # del temp path
        else:
            # count line num
            for j in tqdm(self.files, ncols=0, desc="Counting lines"):
                header = self.header
                with open(j) as f:
                    for line in f:
                        if header:
                            self.header_data = line
                            header = False
                            continue
                        self.line_num += 1
            # train and val line num
            trainset_num = int(self.line_num * self.train_ratio)
            # read and write directly
            F = open(self.save_trainset_name, "w")
            if self.header:
                F.write(self.header_data)
            n = 0
            for i in tqdm(self.files, ncols=0, desc="Write to train and val"):
                header = self.header
                with open(i) as f:
                    for text in f:
                        if header:
                            header = False
                            continue
                        n += 1
                        F.write(text)
                        if n == trainset_num:
                            n = 0
                            F = open(self.save_valset_name, "w")
                            if self.header:
                                F.write(self.header_data)

        end_time = time.time()
        logger.info("Time cost：%ss" % (end_time - start_time))
