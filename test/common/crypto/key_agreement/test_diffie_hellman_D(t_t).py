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


# import json
# import time
# from concurrent import futures

# import grpc

# from common.utils.grpc_channel_options import options
# from common.communication.gRPC.python import trainer_pb2_grpc
# from service.trainer import TrainerService
# from fed_api import Commu
# from fed_api import FedNode
# from fed_api import DataPool
# from fed_api import DiffieHellman


# FedNode.init_fednode()
# FedNode.config["node_id"] = 'node-2'
# FedNode.node_id = 'node-2'

# server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
# trainer_pb2_grpc.add_TrainerServicer_to_server(TrainerService(), server)
# FedNode.add_server(server, "trainer")
# server.start()

# with open("./config/data_pool_config.json") as f:
#     data_pool_config = json.load(f)
#     DataPool(data_pool_config)

# Commu(FedNode.config)

# time.sleep(5)

# dh = DiffieHellman(fed_ids=['node-1', 'node-2'], key_bitlength=3072, optimized=True, channel_name="diffie_hellman")
# secret = dh.exchange()
# print(secret)
# print(dh)