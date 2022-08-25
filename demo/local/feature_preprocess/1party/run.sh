#!/bin/sh

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

if [ "$(uname)" = "Darwin" ]; then
  export PROJECT_HOME=$(greadlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
elif [ "$(uname -s)" = "Linux" ]; then
  export PROJECT_HOME=$(readlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
fi

export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME/python:$PROJECT_HOME/python/common/communication/gRPC/python

datapath="${PROJECT_HOME}/dataset"
if [ ! -d "${PROJECT_HOME}/dataset/breast_cancer_wisconsin_vertical/2party" ]; then
  if [ ! -f "${PROJECT_HOME}/python/xfl.py" ]; then
    python "${PROJECT_HOME}/common/dataset/breast_cancer_wisconsin.py" --mode "vertical" --splits 2 --party "labeled" "1"
  else
    python "${PROJECT_HOME}/python/common/dataset/breast_cancer_wisconsin.py" --mode "vertical" --splits 2 --party "labeled" "1"
  fi
fi

python3 - "$@" <<-END
import pandas as pd
import os
import numpy as np
dir_path = os.path.join(os.environ["PROJECT_HOME"], "dataset")
data_path = os.path.join(dir_path, "breast_cancer_wisconsin_vertical/2party/breast_cancer_wisconsin_vertical_labeled_train.csv")
tmp = pd.read_csv(data_path,index_col=False)
col = tmp.columns[2:]
tmp.iloc[0:4,2:]=pd.DataFrame([[np.NaN]*(len(tmp.columns)-2)]*4,columns=col)
tmp.to_csv(data_path, index=False)
END

type="local"
operator="feature_preprocess"
party="1party"
code="${type}.${operator}.${party}"
config_path="${PROJECT_HOME}/demo/${type}/${operator}/${party}/config"

if [ ! -f "${PROJECT_HOME}/python/xfl.py" ]; then
  EXECUTE_PATH=${PROJECT_HOME}/xfl.py
else
  EXECUTE_PATH=${PROJECT_HOME}/python/xfl.py
fi

cd $PROJECT_HOME
python "$EXECUTE_PATH" -s --config_path ${config_path} &
sleep 1
python "$EXECUTE_PATH" -t node-1 --config_path ${config_path} &
sleep 1
python "$EXECUTE_PATH" -c start --config_path ${config_path} &
