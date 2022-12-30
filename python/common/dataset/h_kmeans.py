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

from sklearn.datasets import make_blobs
import pandas as pd
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", type=int, default=2,
        help="number of parties"
    )
    parser.add_argument(
        "--ndims", type=int, default=2,
        help="number of data dims"
    )
    parser.add_argument(
        "--nsamples", type=int, default=150,
        help="number of samples"
    )

    args = parser.parse_args()

    dirpath = os.path.join(os.environ['PROJECT_HOME'], 'dataset')
    data_folder = os.path.join(
        dirpath, "horizontal_kmeans", f"{args.splits}party")
    print(f"data folder: {data_folder}")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    total_samples = args.nsamples * args.splits + 50

    X, y = make_blobs(
        n_samples=total_samples,
        n_features=args.ndims,
        centers=3,
        random_state=42,
        cluster_std=1.0
    )
    print("Generating dataset")

    for party in range(args.splits):
        start_ind = party * args.nsamples
        end_ind = (party + 1) * args.nsamples
        data_X = X[start_ind:end_ind, :]
        data_y = y[start_ind:end_ind]
        data_X_df = pd.DataFrame(
            data_X, columns=["X_" + str(ind) for ind in range(args.ndims)])
        data_y_df = pd.DataFrame(data_y, columns=["label"])
        data_df = pd.concat([data_y_df, data_X_df], axis=1)
        data_path = os.path.join(data_folder, f"blob_{party+1}.csv")
        data_df.to_csv(data_path, index=False)
        print(f"Writing to {data_path}")

    data_X = X[-50:]
    data_y = y[-50:]
    data_X_df = pd.DataFrame(
        data_X, columns=["X_" + str(ind) for ind in range(args.ndims)])
    data_y_df = pd.DataFrame(data_y, columns=["label"])
    data_df = pd.concat([data_y_df, data_X_df], axis=1)
    data_path = os.path.join(data_folder, f"blob_assist_trainer.csv")
    data_df.to_csv(data_path, index=False)
