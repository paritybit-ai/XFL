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



from common.communication.gRPC.python import control_pb2, status_pb2

from service import fed_control
from service.fed_node import FedNode


def test_trainer_control(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='56001')
    mocker.patch("service.fed_control.trainer_pb2_grpc.TrainerStub.__init__", side_effect=lambda x:None)
    mocker.patch("service.fed_control.trainer_pb2_grpc.TrainerStub.control", create=True, return_value=control_pb2.ControlResponse(code=0,message='test\n'))
    mocker.patch.object(FedNode, 'trainers', {"node-1":"test"})
   

    resp = fed_control.trainer_control(control_pb2.STOP)

    assert resp.message == "STOP Trainer: node-1 Successful.\n"
    
    mocker.patch("service.fed_control.trainer_pb2_grpc.TrainerStub.control", create=True, return_value=control_pb2.ControlResponse(code=1,message='test\n'))
    resp = fed_control.trainer_control(control_pb2.START)
    assert resp.message == "START Trainer: node-1 Failed.\n"


def test_get_trainer_status(mocker):
    mocker.patch.object(FedNode, "create_channel", return_value='56001')
    mocker.patch.object(FedNode, 'trainers', {"node-1":"test"})
    mocker.patch("service.fed_control.trainer_pb2_grpc.TrainerStub.__init__", side_effect=lambda x:None)
    mocker.patch("service.fed_control.trainer_pb2_grpc.TrainerStub.status", create=True, return_value=status_pb2.StatusResponse(trainerStatus={"node-1":status_pb2.Status(code=1,status='IDLE')}))


    resp = fed_control.get_trainer_status()
    assert resp == {"node-1":status_pb2.Status(code=1,status='IDLE')}
