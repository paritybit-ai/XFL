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
import unittest.mock as um

import grpc

from common.utils.grpc_channel_options import insecure_options
from service.fed_node import FedNode


class mock_server():
    def add_insecure_port(self, address):
        self.address = address


class Test_FedNode():
    def test_init_fednode(self, monkeypatch):
        task_network = '''{
            "nodes": {
                "node-1": {
                    "endpoints": [
                        {
                            "fuwuEndpointId": "scheduler-endpoint-1",
                            "url": "localhost:55001"
                        },
                        {
                            "fuwuEndpointId": "trainer-endpoint-1",
                            "url": "localhost:56001"
                        },
                        {
                            "fuwuEndpointId": "assist-trainer-endpoint-1",
                            "url": "localhost:57001"
                        }
                    ],
                    "name": "promoter"
                }
            }
        }'''
        monkeypatch.setenv("__ENIGMA_FEDAPP_LOCAL_TASK_NODE_ID__", "node-1")
        monkeypatch.setenv("__ENIGMA_FEDAPP_TASK_NETWORK__", task_network)
        monkeypatch.setenv(
            "DEBUG_LISTENING_PORT", '{"scheduler": 55001, "assist_trainer":57001, "node-1": 56001}')

        FedNode.init_fednode("scheduler", "scheduler")
        assert FedNode.node_id == "scheduler"
        assert FedNode.listening_port == 55001

        FedNode.init_fednode("assist_trainer", "assist_trainer")
        assert FedNode.node_id == "assist_trainer"
        assert FedNode.listening_port == 57001

        FedNode.init_fednode("trainer", "node-1")
        assert FedNode.config == {"node_id": "node-1", "scheduler": {"node_id": "node-1", "host": "localhost", "port": "55001", "use_tls": False}, "trainer": {
            "node-1": {"host": "localhost", "port": "56001", "use_tls": False}, "assist_trainer": {"host": "localhost", "port": "57001", "use_tls": False}}}
        assert FedNode.scheduler_host == "localhost"
        assert FedNode.scheduler_port == "55001"
        assert FedNode.trainers == {"node-1": {"host": "localhost", "port": "56001", "use_tls": False},
                                    "assist_trainer": {"host": "localhost", "port": "57001", "use_tls": False}}
        assert FedNode.listening_port == 56001

    def test_init_fednode2(self, tmp_path):
        path = tmp_path / "fed_conf_scheduler.json"
        fed_conf = {
            "fed_info": {
                "scheduler": {
                    "node-1": "localhost:55001"
                },
                "trainer": {
                    "node-1": "localhost:56001",
                    "node-2": "localhost:56002"
                },
                "assist_trainer": {
                    "assist_trainer": "localhost:57001"
                }
            },
            "redis_server": "localhost:6379",
            "grpc": {
                "use_tls": False
            }
        }
        f = open(path, 'w')
        json.dump(fed_conf, f)
        f.close()
        FedNode.init_fednode("scheduler", "scheduler", tmp_path)
        assert FedNode.node_id == "scheduler"
        assert FedNode.listening_port == '55001'

        path2 = tmp_path / "fed_conf.json"
        f = open(path2, 'w')
        json.dump(fed_conf, f)
        f.close()
        FedNode.init_fednode("assist_trainer", "assist_trainer", tmp_path)
        assert FedNode.node_id == "assist_trainer"
        assert FedNode.listening_port == '57001'

        FedNode.init_fednode("trainer", "node-1", tmp_path)
        assert FedNode.config == {'node_id': 'node-1', 'scheduler': {'node_id': 'node-1', 'host': 'localhost', 'port': '55001', 'use_tls': False}, 
                                  'trainer': {'assist_trainer': {'node_id': 'assist_trainer', 'host': 'localhost', 'port': '57001', 'use_tls': False}, 'node-1': {'host': 'localhost', 'port': '56001', 'use_tls': False}, 'node-2': {'host': 'localhost', 'port': '56002', 'use_tls': False}}, 'redis_server': {'host': 'localhost', 'port': '6379'}}
        assert FedNode.scheduler_host == "localhost"
        assert FedNode.scheduler_port == "55001"
        assert FedNode.trainers == {"node-1": {"host": "localhost", "port": "56001", "use_tls": False},
                                    'node-2': {'host': 'localhost', 'port': '56002', 'use_tls': False},
                                    "assist_trainer": {"host": "localhost", 'node_id': 'assist_trainer', "port": "57001", "use_tls": False}}
        assert FedNode.listening_port == '56001'

    def test_add_server(self, mocker):
        server = mock_server()
        mocker.patch.object(FedNode, 'listening_port', 55001)
        spy_add_server = mocker.spy(FedNode, 'add_server')
        FedNode.add_server(server)
        assert server.address == "[::]:55001"

    def test_create_channel(self, mocker):
        mocker.patch.object(FedNode, 'scheduler_host', "localhost")
        mocker.patch.object(FedNode, 'scheduler_port', "55001")
        mocker.patch.object(FedNode, 'config', {"scheduler": {
                            "node_id": "node-1", "host": "localhost", "port": "55001", "use_tls": False}})
        mocker.patch.object(FedNode, 'trainers', {"node-1": {"host": "localhost", "port": "56001",
                            "use_tls": True}, "assist_trainer": {"host": "localhost", "port": "57001", "use_tls": False}})
        mocker.patch("grpc.secure_channel", return_value="secure_channel")
        mocker.patch("grpc.insecure_channel", return_value="insecure_channel")
        mocker.patch("grpc.intercept_channel",
                     return_value='intercept_channel')

        channel = FedNode.create_channel("node-1")
        assert FedNode.channels["node-1"] == "intercept_channel"

        channel = FedNode.create_channel("scheduler")
        assert FedNode.channels["scheduler"] == "intercept_channel"
        grpc.insecure_channel.assert_called_once_with(
            "localhost:55001", options=insecure_options)

    def test_load_root_certificates(self, mocker):
        mocker.patch("os.getcwd", return_value=os.path.join(
            os.getcwd(), 'python'))
        mocker.patch('builtins.open', um.mock_open(read_data=b"1"))
        root_certificates = FedNode.load_root_certificates()
        assert root_certificates == b"11"

    def test_load_client_cert(self, mocker):
        mocker.patch.object(FedNode, "config", {
                            "cert": {"client.key": "test", "client.crt": "test", "ca.crt": "test"}})
        mocker.patch('builtins.open', um.mock_open(read_data='test'))
        private_key, certificate_chain, root_certificates = FedNode.load_client_cert()
        assert private_key == 'test'
        assert certificate_chain == 'test'
        assert root_certificates == 'test'
