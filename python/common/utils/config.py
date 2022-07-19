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


def load_json_config(file):
    with open(file) as json_data_file:
        return json.load(json_data_file)


def get_str_config(s):
    return json.loads(s)


def parse_config(s):
    json_str = json.loads(s)
    config = {
        "scheduler": {},
        "trainer": {}
    }

    for node_id in json_str["nodes"]:
        for endpoint in json_str["nodes"][node_id]["endpoints"]:
            url = endpoint["url"]
            if "grpcs://" in url:
                use_tls = True
                url = url.replace("grpcs://", "")
            else:
                use_tls = False
                url = url.replace("grpc://", "")
            host = url.split(":")[0]
            port = url.split(":")[1]
            if "scheduler" in endpoint["fuwuEndpointId"]:
                config["scheduler"]["node_id"] = node_id
                config["scheduler"]["host"] = host
                config["scheduler"]["port"] = port
                config["scheduler"]["use_tls"] = use_tls
            elif "assist-trainer" in endpoint["fuwuEndpointId"]:
                config["trainer"]["assist_trainer"] = {}
                config["trainer"]["assist_trainer"]["host"] = host
                config["trainer"]["assist_trainer"]["port"] = port
                config["trainer"]["assist_trainer"]["use_tls"] = use_tls
            elif "trainer" in endpoint["fuwuEndpointId"]:
                config["trainer"][node_id] = {}
                config["trainer"][node_id]["host"] = host
                config["trainer"][node_id]["port"] = port
                config["trainer"][node_id]["use_tls"] = use_tls
    return config


def refill_config(custom_conf: dict, default_conf: dict):
    """fill custom_conf by default_conf if a key is missing in custom_conf iteratively"""
    for k, v in default_conf.items():
        if k not in custom_conf:
            custom_conf[k] = v
        else:
            if isinstance(v, dict):
                custom_conf[k] = refill_config(custom_conf[k], v)
    return custom_conf