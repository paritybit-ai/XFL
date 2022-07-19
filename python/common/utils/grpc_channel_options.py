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


secure_options = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ('grpc.enable_retries', 1),
    ('grpc.service_config',
     '{"retryPolicy":{ "maxAttempts": 4, "initialBackoff": "0.01s", "maxBackoff": "0.01s", "backoffMutiplier": 1, "retryableStatusCodes": ["UNAVAILABLE"]}}')
]

insecure_options = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ('grpc.enable_retries', 1),
    ('grpc.service_config', '{"retryPolicy":{ "maxAttempts": 4, "initialBackoff": "0.01s", "maxBackoff": "0.01s", "backoffMutiplier": 1, "retryableStatusCodes": ["UNAVAILABLE"]}}')
]
