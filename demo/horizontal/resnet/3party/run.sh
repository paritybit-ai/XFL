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


if [ "$(uname)" = "Darwin" ]
then
  export PROJECT_HOME=$(greadlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
elif [ "$(uname -s)" = "Linux" ]
then
  export PROJECT_HOME=$(readlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
fi

export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME/python:$PROJECT_HOME/python/common/communication/gRPC/python

datapath="${PROJECT_HOME}/dataset"
if [ ! -d "${PROJECT_HOME}/dataset/cifar-10/3party" ]; then
   python "${PROJECT_HOME}/python/common/dataset/cifar.py" --splits 3 --party "1" "2" "3" --keep_raw_data
fi

type="horizontal"
operator="resnet"
party="3party"
code="${type}.${operator}.${party}"
config_path="${PROJECT_HOME}/demo/${type}/${operator}/${party}/config"

if [ ! -f "${PROJECT_HOME}/python/xfl.py" ]; then
  EXECUTE_PATH=${PROJECT_HOME}/xfl.py
else
  EXECUTE_PATH=${PROJECT_HOME}/python/xfl.py
fi

cd $PROJECT_HOME
python "${PROJECT_HOME}/python/xfl.py" -t node-1 --config_path ${config_path} &
sleep 1
python "${PROJECT_HOME}/python/xfl.py" -t node-2 --config_path ${config_path} &
sleep 1
python "${PROJECT_HOME}/python/xfl.py" -t node-3 --config_path ${config_path} &
sleep 1
python "${PROJECT_HOME}/python/xfl.py" -a --config_path ${config_path} &
sleep 1
python "${PROJECT_HOME}/python/xfl.py" -s --config_path ${config_path} &
sleep 1
python "${PROJECT_HOME}/python/xfl.py" -c start --config_path ${config_path} &