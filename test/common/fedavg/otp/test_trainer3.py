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
# from fed_api import get_fedavg_trainer_inst
# from random_input import param_torch, param_numpy, weight_factors, sec_conf


# def do_fedavg(id, sec_conf):
#     fedavg_trainer = get_fedavg_trainer_inst(sec_conf)

#     if 'torch' in sec_conf['data_type']:
#         local_weight = param_torch[id-1]
#     elif 'numpy' in sec_conf['data_type']:
#         local_weight = param_numpy[id-1]
        
#     weight_factor = weight_factors[id-1]
#     fedavg_trainer.aggregate(local_weight, weight_factor)


# if __name__ == "__main__":

#     id = 'node-3'

#     FedNode.init_fednode()
#     FedNode.config["node_id"] = str(id)
#     FedNode.node_id = str(id)

#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
#     trainer_pb2_grpc.add_TrainerServicer_to_server(TrainerService(), server)
#     FedNode.add_server(server, "trainer")
#     server.start()

#     with open("./config/data_pool_config.json") as f:
#         data_pool_config = json.load(f)
#         DataPool(data_pool_config)

#     Commu(FedNode.config)

#     time.sleep(5)

#     for conf in sec_conf:
#         do_fedavg(3, conf)

