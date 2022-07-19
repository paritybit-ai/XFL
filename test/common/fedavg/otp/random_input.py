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


# import numpy as np
# import torch
# import random
# from collections import OrderedDict

# seed = 0
# torch.manual_seed(seed)
# # torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# # torch.backends.cudnn.deterministic = True


# def gen_params(dtype: str):
#     shape_dict = OrderedDict(
#         {
#             'a': (2, 3, 4),
#             'b': (5),
#             'c': ()
#         }
#     )
    
#     w = OrderedDict()

#     for k, v in shape_dict.items():
#         if dtype == 'numpy':
#             w[k] = np.random.random(v).astype(np.float32) * 2 - 1
#         elif dtype == 'torch':
#             w[k] = torch.rand(v) * 2 - 1
            
#     return w


# num_trainer = 3

# param_torch = [gen_params('torch') for i in range(num_trainer)]
# param_numpy = [gen_params('numpy') for i in range(num_trainer)]

# weight_factors = [random.random() for i in range(num_trainer)]

# sec_conf = [
#     {
#         "method": "otp",
#         "key_bitlength": 128,
#         "data_type": "torch.Tensor",
#         "key_exchange": {
#             "key_bitlength": 3072,
#             "optimized": True
#         },
#         "csprng": {
#             "name": "hmac_drbg",
#             "method": "sha512",
#         }
#     },
#     {
#         "method": "otp",
#         "key_bitlength": 128,
#         "data_type": "numpy.ndarray",
#         "key_exchange": {
#             "key_bitlength": 3072,
#             "optimized": True
#         },
#         "csprng": {
#             "name": "hmac_drbg",
#             "method": "sha512",
#         }
#     }
# ]

