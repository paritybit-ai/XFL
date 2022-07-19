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



import json

from common.utils.config import parse_config, refill_config


def test_parse_config(mocker):
    json_str = json.dumps(
    {
        "nodes": {
            "trainer": {
                "endpoints": [
                    {
                        "fuwuEndpointId": "trainer-endpoint-2",
                        "url": "grpcs://localhost:56002",
                    }
                ],
                "name": "follower"
            },
            "label_trainer": {
                "endpoints": [
                    {
                        "fuwuEndpointId": "assist-trainer-endpoint-1",
                        "url": "localhost:56001",
                    },
                    {
                        "fuwuEndpointId": "scheduler-endpoint-1",
                        "url": "localhost:55001",
                    }
                ],
                "name": "master"
            }
        }
    }
    )

    config = parse_config(json_str)
    assert config == {'scheduler': {'node_id': 'label_trainer', 'host': 'localhost', 'port': '55001', 'use_tls': False}, 'trainer': {'trainer': {
        'host': 'localhost', 'port': '56002', 'use_tls': True}, 'assist_trainer': {'host': 'localhost', 'port': '56001', 'use_tls': False}}}


def test_refill_config():
    custom_conf = {"1":{}}
    default_conf = {"2":2, "1":{"3":3}}
    config = refill_config(custom_conf, default_conf)

    assert config == {"2":2, "1":{"3":3}}
