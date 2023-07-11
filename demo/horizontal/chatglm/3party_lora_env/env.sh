#! /bin/bash

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


# shellcheck disable=SC2155

# source env.sh
# brew install coreutils

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
export ENIGMA_redis_HOST=localhost
export __ENIGMA_FEDAPP_LOCAL_TASK_NODE_ID__=node-1
export DEBUG_LISTENING_PORT='{"scheduler": 55004, "node-1": 56001, "node-2": 56002,"assist_trainer": 55003}'
export __ENIGMA_FEDAPP_TASK_NETWORK__='{
    "nodes": {
        "node-2": {
            "endpoints": [
                {
                    "fuwuEndpointId": "trainer-endpoint-2",
                    "url": "localhost:56002"
                }
            ],
            "name": "follower2"
        },
        "node-1": {
            "endpoints": [
                
                {
                    "fuwuEndpointId": "trainer-endpoint-1",
                    "url": "localhost:56001"
                },
                {
                    "fuwuEndpointId": "scheduler-endpoint-1",
                    "url": "localhost:55004"
                },
                {
                    "fuwuEndpointId": "assist-trainer-endpoint-1",
                    "url": "localhost:55003"
                }
            ],
            "name": "follower1"
        }
    }
}'
