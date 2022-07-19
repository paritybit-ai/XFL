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
# from typing import OrderedDict
# from functools import reduce

# import grpc
# import numpy as np
# import torch

# from common.utils.grpc_channel_options import options
# from common.communication.gRPC.python import scheduler_pb2_grpc
# from service.scheduler import SchedulerService
# from fed_api import Commu
# from fed_api import FedNode
# from fed_api import DataPool
# from fed_api import get_fedavg_scheduler_inst
# from random_input import param_torch, param_numpy, weight_factors, sec_conf


# def almost_equal(a, b):
#     for k in a:
#         if isinstance(a[k], np.ndarray):
#             return np.all(a[k] - b[k] < 1e-4)
#         else:
#             return torch.all(a[k] - b[k] < 1e-4)
        
        
# def do_fedavg(sec_conf):
#     fedavg_trainer = get_fedavg_scheduler_inst(sec_conf)

#     result = fedavg_trainer.aggregate(weight_factors)

#     def f(x, y):
#         for k in x:
#             x[k] += y[k]
#         return x
    
#     if 'torch' in sec_conf["data_type"]:
#         param = param_torch
#     elif 'numpy' in sec_conf["data_type"]:
#         param = param_numpy

#     for i, item in enumerate(param):
#         for k in item:
#             param[i][k] *= weight_factors[i]

#     expected_result = reduce(f, param)

#     sum_weight_factors = sum(weight_factors)
#     for k in expected_result:
#         expected_result[k] /= sum_weight_factors

#     assert almost_equal(result, expected_result)
   

# if __name__ == "__main__":
#     FedNode.init_fednode(is_scheduler=True)
#     FedNode.config["node_id"] = 'scheduler'
#     FedNode.node_id = 'scheduler'

#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
#     scheduler_pb2_grpc.add_SchedulerServicer_to_server(SchedulerService(), server)
#     FedNode.add_server(server, "scheduler")
#     server.start()

#     with open("./config/data_pool_config.json") as f:
#         data_pool_config = json.load(f)
#         DataPool(data_pool_config)

#     Commu(FedNode.config)

#     time.sleep(5)

#     for conf in sec_conf:
#         do_fedavg(conf)
