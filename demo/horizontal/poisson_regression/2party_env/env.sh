#! /bin/bash
# shellcheck disable=SC2155

# source env.sh
# brew install coreutils

if [ "$(uname)" = "Darwin" ]
then
  export PROJECT_HOME=$(greadlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
elif [ "$(uname -s)" == "Linux" ]
then
  export PROJECT_HOME=$(readlink -f ../../../../)
  echo "PROJECT_HOME:""$PROJECT_HOME"
fi

export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME/python:$PROJECT_HOME/python/common/communication/gRPC/python
export ENIGMA_redis_HOST=localhost
export __ENIGMA_FEDAPP_LOCAL_TASK_NODE_ID__=node-1
export DEBUG_LISTENING_PORT='{"scheduler": 55001, "assist_trainer": 57001, "node-1": 56001, "node-2": 56002}'
export __ENIGMA_FEDAPP_TASK_NETWORK__='{
    "nodes": {
        "node-1": {
            "endpoints": [
                {
                    "fuwuEndpointId": "scheduler-endpoint-1",
                    "url": "localhost:55001"
                },
                {
                    "fuwuEndpointId": "assist-trainer-endpoint-1",
                    "url": "localhost:57001"
                },
                {
                    "fuwuEndpointId": "trainer-endpoint-1",
                    "url": "localhost:56001"
                }
            ],
            "name": "promoter"
        },
        "node-2": {
            "endpoints": [
                {
                    "fuwuEndpointId": "trainer-endpoint-1",
                    "url": "localhost:56002"
                }
            ],
            "name": "provider1"
        }
    }
}'
