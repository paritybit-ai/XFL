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


from common.utils.logger import logger


class FedConfParser():
    @classmethod
    def parse_dict_conf(cls, conf: dict, node_id: str = ''):
        if "node_id" not in conf or conf.get("node_id") == '':
            conf["node_id"] = node_id
        else:
            if node_id != conf["node_id"]:
                logger.warning(f"The input node_id {node_id} and node_id {conf['node_id']}in fed_conf.json not the same, use input node_id.")
                
        out_conf = {}
        out_conf["node_id"] = conf["node_id"]
        
        grpc_conf = conf.get("grpc")
        if grpc_conf is None:
            use_tls = False
        else:
            use_tls = grpc_conf.get("use_tls") or False
            
        fed_info = conf.get("fed_info")
        scheduler_conf = fed_info["scheduler"]
        scheduler_node_id = list(scheduler_conf.keys())[0]
        scheduler_host, scheduler_port = scheduler_conf[scheduler_node_id].replace(" ", "").split(":")
        
        out_conf["scheduler"] = {
            "node_id": scheduler_node_id,
            "host": scheduler_host,
            "port": scheduler_port,
            "use_tls": use_tls
        }
        
        out_conf["trainer"] = {}
        
        if "assist_trainer" in fed_info:
            node_id = list(fed_info["assist_trainer"].keys())[0]
            host, port = fed_info["assist_trainer"][node_id].replace(" ", "").split(":")
            out_conf["trainer"]["assist_trainer"] = {
                "node_id": node_id,
                "host": host,
                "port": port,
                "use_tls": use_tls
            }
        
        for node_id, host_port in fed_info["trainer"].items():
            host, port = host_port.replace(" ", "").split(":")
            out_conf["trainer"][node_id] = {
                "host": host,
                "port": port,
                "use_tls": use_tls
            }
            
        host, port = conf["redis_server"].replace(" ", "").split(":")
        out_conf["redis_server"] = {
            "host": host,
            "port": port
        }
        return out_conf
