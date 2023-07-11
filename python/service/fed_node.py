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
import os
from pathlib import Path
from typing import Callable, Any

import grpc
from grpc_interceptor import ClientCallDetails, ClientInterceptor

from common.utils.config import get_str_config, parse_config
from common.utils.fed_conf_parser import FedConfParser
from common.utils.grpc_channel_options import insecure_options, secure_options


class FedNode(object):
    config = {}
    node_id = ""
    scheduler_host = ""
    scheduler_port = ""
    redis_host = ""
    redis_port = ""
    trainers = {}
    channels = {}
    listening_port = None

    @classmethod
    def init_fednode(cls, identity: str = "scheduler", debug_node_id: str = "scheduler", conf_dir: str = ''):
        path1 = Path(conf_dir, "fed_conf.json")
        path2 = Path(conf_dir, "fed_conf_"+debug_node_id+'.json')
        
        if os.path.exists(path1):
            path = path1
        elif os.path.exists(path2):
            path = path2
        else:
            path = ''
        
        if path == '':
            if identity == "scheduler":
                cls.node_id = "scheduler"
                cls.listening_port = 55001
            elif identity == "assist_trainer":
                cls.node_id = "assist_trainer"
                cls.listening_port = 57001
            else:
                cls.node_id = os.getenv("__ENIGMA_FEDAPP_LOCAL_TASK_NODE_ID__")
                cls.listening_port = 56001
            cls.config = parse_config(os.getenv("__ENIGMA_FEDAPP_TASK_NETWORK__"))
            cls.config["node_id"] = cls.node_id
            
            for name in cls.config["trainer"]:
                if cls.node_id == name:
                    cls.node_name = cls.config["trainer"][cls.node_id]["name"]
                    
            if not hasattr(cls, "node_name"):
                for name in cls.config["scheduler"]:
                    if cls.node_id == name:
                        cls.node_name = cls.config["scheduler"][cls.node_id]["name"]

            cls.scheduler_host = cls.config["scheduler"]["host"]
            cls.scheduler_port = cls.config["scheduler"]["port"]
            cls.trainers = cls.config["trainer"]
            if os.getenv("DEBUG_LISTENING_PORT") is not None:
                cls.node_id = debug_node_id
                cls.config["node_id"] = debug_node_id
                cls.listening_port = get_str_config(os.getenv("DEBUG_LISTENING_PORT"))[debug_node_id]
                
            cls.redis_host = os.getenv("ENIGMA_redis_HOST")
            cls.redis_port = '6379'
        else:
            with open(path, 'r') as f:
                conf_dict = json.load(f)
                cls.config = FedConfParser.parse_dict_conf(conf_dict, debug_node_id)
                cls.node_id = cls.config["node_id"]
                cls.node_name = cls.config["node_id"]
                
                for node_id in cls.config["trainer"]:
                    cls.config["trainer"][node_id]["name"] = node_id
                
                if identity == "scheduler":
                    cls.listening_port = cls.config["scheduler"]["port"]
                else:
                    for node_id in cls.config["trainer"]:
                        if node_id == "assist_trainer":
                            if cls.node_id == cls.config["trainer"]["assist_trainer"]["node_id"]:
                                cls.listening_port = cls.config["trainer"]["assist_trainer"]["port"]
                                break
                        elif cls.node_id == node_id:
                            cls.listening_port = cls.config["trainer"][node_id]["port"]
                            break
                
            cls.scheduler_host = cls.config["scheduler"]["host"]
            cls.scheduler_port = cls.config["scheduler"]["port"]
            cls.trainers = cls.config["trainer"]
            cls.redis_host = cls.config["redis_server"]["host"]
            cls.redis_port = cls.config["redis_server"]["port"]

    @classmethod
    def add_server(cls, server):
        server.add_insecure_port(f"[::]:{cls.listening_port}")

    @classmethod
    def create_channel(cls, node_id: str):
        if node_id not in cls.channels.keys():
            if node_id == "scheduler":
                host = cls.scheduler_host
                port = cls.scheduler_port
                use_tls = cls.config["scheduler"]["use_tls"]
            else:
                host = cls.trainers[node_id]["host"]
                port = cls.trainers[node_id]["port"]
                use_tls = cls.trainers[node_id]["use_tls"]

            addr_list = port.split("/")
            port = addr_list[0]
            sub_addr = '/'.join(addr_list[1:])

            if use_tls:
                root_certificates = cls.load_root_certificates()
                credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
                channel = grpc.secure_channel(f"{host}:{port}", credentials, options=secure_options)
            else:
                channel = grpc.insecure_channel(f"{host}:{port}", options=insecure_options)
            
            class ClientPathInterceptor(ClientInterceptor):
                def intercept(
                        self,
                        method: Callable,
                        request_or_iterator: Any,
                        call_details: grpc.ClientCallDetails):
                    path_list = call_details.method.split("/")
                    path_list.insert(1, sub_addr)
                    new_method = '/' + os.path.join(*path_list)

                    new_call_details = ClientCallDetails(
                        new_method,
                        call_details.timeout, call_details.metadata,
                        call_details.credentials, call_details.wait_for_ready,
                        call_details.compression)
                    return method(request_or_iterator, new_call_details)
            channel = grpc.intercept_channel(channel, ClientPathInterceptor())
            cls.channels[node_id] = channel

        return cls.channels[node_id]
    
    @classmethod
    def init_job_id(cls):
        if cls.rs.get("XFL_JOB_ID") is None:
            cls.rs.set("XFL_JOB_ID", 0)

    @classmethod
    def load_root_certificates(cls):
        ca_file = os.getcwd() + "/common/certificates/ca.crt"
        ca_bundle_file = os.getcwd() + "/common/certificates/ca-bundle.crt"
        root_certificates = b""
        if os.path.isfile(ca_file):
            with open(ca_file, "rb") as f:
                root_certificates += f.read()
        if os.path.isfile(ca_bundle_file):
            with open(ca_bundle_file, "rb") as f:
                root_certificates += f.read()

        if root_certificates == b"":
            return None
        else:
            return root_certificates

    @classmethod
    def load_client_cert(cls):
        with open(cls.config["cert"]["client.key"], "rb") as f:
            private_key = f.read()
        with open(cls.config["cert"]["client.crt"], "rb") as f:
            certificate_chain = f.read()
        with open(cls.config["cert"]["ca.crt"], "rb") as f:
            root_certificates = f.read()
        return private_key, certificate_chain, root_certificates

