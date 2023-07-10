# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: control.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rcontrol.proto\x12\x07\x63ontrol\"5\n\x0e\x43ontrolRequest\x12#\n\x07\x63ontrol\x18\x01 \x01(\x0e\x32\x12.control.Operation\".\n\x0bNodeLogPath\x12\x0e\n\x06nodeId\x18\x01 \x01(\t\x12\x0f\n\x07logPath\x18\x02 \x01(\t\"D\n\x10StageNodeLogPath\x12\x0f\n\x07stageId\x18\x01 \x01(\x05\x12\x0e\n\x06nodeId\x18\x02 \x01(\t\x12\x0f\n\x07logPath\x18\x03 \x01(\t\"\xba\x01\n\x0f\x43ontrolResponse\x12\r\n\x05jobId\x18\x01 \x01(\x05\x12\x0c\n\x04\x63ode\x18\x02 \x01(\x05\x12\x0f\n\x07message\x18\x03 \x01(\t\x12\x19\n\x11\x64umpedTrainConfig\x18\x04 \x01(\t\x12)\n\x0bnodeLogPath\x18\x05 \x03(\x0b\x32\x14.control.NodeLogPath\x12\x33\n\x10stageNodeLogPath\x18\x06 \x03(\x0b\x32\x19.control.StageNodeLogPath*F\n\tOperation\x12\r\n\tOPERATION\x10\x00\x12\t\n\x05START\x10\x01\x12\x08\n\x04STOP\x10\x02\x12\t\n\x05PAUSE\x10\x03\x12\n\n\x06UPDATE\x10\x04\x62\x06proto3')

_OPERATION = DESCRIPTOR.enum_types_by_name['Operation']
Operation = enum_type_wrapper.EnumTypeWrapper(_OPERATION)
OPERATION = 0
START = 1
STOP = 2
PAUSE = 3
UPDATE = 4


_CONTROLREQUEST = DESCRIPTOR.message_types_by_name['ControlRequest']
_NODELOGPATH = DESCRIPTOR.message_types_by_name['NodeLogPath']
_STAGENODELOGPATH = DESCRIPTOR.message_types_by_name['StageNodeLogPath']
_CONTROLRESPONSE = DESCRIPTOR.message_types_by_name['ControlResponse']
ControlRequest = _reflection.GeneratedProtocolMessageType('ControlRequest', (_message.Message,), {
  'DESCRIPTOR' : _CONTROLREQUEST,
  '__module__' : 'control_pb2'
  # @@protoc_insertion_point(class_scope:control.ControlRequest)
  })
_sym_db.RegisterMessage(ControlRequest)

NodeLogPath = _reflection.GeneratedProtocolMessageType('NodeLogPath', (_message.Message,), {
  'DESCRIPTOR' : _NODELOGPATH,
  '__module__' : 'control_pb2'
  # @@protoc_insertion_point(class_scope:control.NodeLogPath)
  })
_sym_db.RegisterMessage(NodeLogPath)

StageNodeLogPath = _reflection.GeneratedProtocolMessageType('StageNodeLogPath', (_message.Message,), {
  'DESCRIPTOR' : _STAGENODELOGPATH,
  '__module__' : 'control_pb2'
  # @@protoc_insertion_point(class_scope:control.StageNodeLogPath)
  })
_sym_db.RegisterMessage(StageNodeLogPath)

ControlResponse = _reflection.GeneratedProtocolMessageType('ControlResponse', (_message.Message,), {
  'DESCRIPTOR' : _CONTROLRESPONSE,
  '__module__' : 'control_pb2'
  # @@protoc_insertion_point(class_scope:control.ControlResponse)
  })
_sym_db.RegisterMessage(ControlResponse)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OPERATION._serialized_start=388
  _OPERATION._serialized_end=458
  _CONTROLREQUEST._serialized_start=26
  _CONTROLREQUEST._serialized_end=79
  _NODELOGPATH._serialized_start=81
  _NODELOGPATH._serialized_end=127
  _STAGENODELOGPATH._serialized_start=129
  _STAGENODELOGPATH._serialized_end=197
  _CONTROLRESPONSE._serialized_start=200
  _CONTROLRESPONSE._serialized_end=386
# @@protoc_insertion_point(module_scope)
