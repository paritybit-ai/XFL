# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: checker.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rchecker.proto\x12\x07\x63hecker\"\x1b\n\x0c\x44ictPathInfo\x12\x0b\n\x03key\x18\x01 \x01(\t\"\x1d\n\x0cListPathInfo\x12\r\n\x05index\x18\x01 \x01(\x05\"l\n\x08PathInfo\x12)\n\x08\x64ictPath\x18\x01 \x01(\x0b\x32\x15.checker.DictPathInfoH\x00\x12)\n\x08listPath\x18\x02 \x01(\x0b\x32\x15.checker.ListPathInfoH\x00\x42\n\n\x08pathInfo\">\n\x08ItemInfo\x12#\n\x08pathInfo\x18\x01 \x03(\x0b\x32\x11.checker.PathInfo\x12\r\n\x05notes\x18\x02 \x01(\t\"N\n\x16\x43rossStagePositionInfo\x12\x0f\n\x07stageId\x18\x01 \x01(\x05\x12#\n\x08pathInfo\x18\x02 \x03(\x0b\x32\x11.checker.PathInfo\"`\n\x12\x43rossStageItemInfo\x12\x13\n\x0b\x64umpedValue\x18\x01 \x01(\t\x12\x35\n\x0cpositionList\x18\x02 \x03(\x0b\x32\x1f.checker.CrossStagePositionInfo\"\x9f\x01\n\x0bStageResult\x12\x0f\n\x07stageId\x18\x01 \x01(\x05\x12\x1b\n\x13\x64umpedCheckedConfig\x18\x02 \x01(\t\x12)\n\x0eunmatchedItems\x18\x03 \x03(\x0b\x32\x11.checker.ItemInfo\x12\x13\n\x0bpassedRules\x18\x04 \x01(\x05\x12\x14\n\x0c\x63heckedRules\x18\x05 \x01(\x05\x12\x0c\n\x04\x63ode\x18\x06 \x01(\x05\"O\n\x10MultiStageResult\x12-\n\x0fstageResultList\x18\x01 \x03(\x0b\x32\x14.checker.StageResult\x12\x0c\n\x04\x63ode\x18\x02 \x01(\x05\"\xca\x01\n\x10\x43rossStageResult\x12:\n\x15\x64uplicatedInputOutput\x18\x01 \x03(\x0b\x32\x1b.checker.CrossStageItemInfo\x12\x35\n\x10\x62lankInputOutput\x18\x02 \x03(\x0b\x32\x1b.checker.CrossStageItemInfo\x12\x35\n\x10nonexistentInput\x18\x03 \x03(\x0b\x32\x1b.checker.CrossStageItemInfo\x12\x0c\n\x04\x63ode\x18\x04 \x01(\x05\"M\n\x16\x43heckTaskConfigRequest\x12\x19\n\x11\x64umpedTrainConfig\x18\x01 \x01(\t\x12\x18\n\x10\x65xistedInputPath\x18\x02 \x03(\t\"\xa2\x01\n\x17\x43heckTaskConfigResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x33\n\x10multiStageResult\x18\x03 \x01(\x0b\x32\x19.checker.MultiStageResult\x12\x33\n\x10\x63rossStageResult\x18\x04 \x01(\x0b\x32\x19.checker.CrossStageResultb\x06proto3')



_DICTPATHINFO = DESCRIPTOR.message_types_by_name['DictPathInfo']
_LISTPATHINFO = DESCRIPTOR.message_types_by_name['ListPathInfo']
_PATHINFO = DESCRIPTOR.message_types_by_name['PathInfo']
_ITEMINFO = DESCRIPTOR.message_types_by_name['ItemInfo']
_CROSSSTAGEPOSITIONINFO = DESCRIPTOR.message_types_by_name['CrossStagePositionInfo']
_CROSSSTAGEITEMINFO = DESCRIPTOR.message_types_by_name['CrossStageItemInfo']
_STAGERESULT = DESCRIPTOR.message_types_by_name['StageResult']
_MULTISTAGERESULT = DESCRIPTOR.message_types_by_name['MultiStageResult']
_CROSSSTAGERESULT = DESCRIPTOR.message_types_by_name['CrossStageResult']
_CHECKTASKCONFIGREQUEST = DESCRIPTOR.message_types_by_name['CheckTaskConfigRequest']
_CHECKTASKCONFIGRESPONSE = DESCRIPTOR.message_types_by_name['CheckTaskConfigResponse']
DictPathInfo = _reflection.GeneratedProtocolMessageType('DictPathInfo', (_message.Message,), {
  'DESCRIPTOR' : _DICTPATHINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.DictPathInfo)
  })
_sym_db.RegisterMessage(DictPathInfo)

ListPathInfo = _reflection.GeneratedProtocolMessageType('ListPathInfo', (_message.Message,), {
  'DESCRIPTOR' : _LISTPATHINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.ListPathInfo)
  })
_sym_db.RegisterMessage(ListPathInfo)

PathInfo = _reflection.GeneratedProtocolMessageType('PathInfo', (_message.Message,), {
  'DESCRIPTOR' : _PATHINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.PathInfo)
  })
_sym_db.RegisterMessage(PathInfo)

ItemInfo = _reflection.GeneratedProtocolMessageType('ItemInfo', (_message.Message,), {
  'DESCRIPTOR' : _ITEMINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.ItemInfo)
  })
_sym_db.RegisterMessage(ItemInfo)

CrossStagePositionInfo = _reflection.GeneratedProtocolMessageType('CrossStagePositionInfo', (_message.Message,), {
  'DESCRIPTOR' : _CROSSSTAGEPOSITIONINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.CrossStagePositionInfo)
  })
_sym_db.RegisterMessage(CrossStagePositionInfo)

CrossStageItemInfo = _reflection.GeneratedProtocolMessageType('CrossStageItemInfo', (_message.Message,), {
  'DESCRIPTOR' : _CROSSSTAGEITEMINFO,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.CrossStageItemInfo)
  })
_sym_db.RegisterMessage(CrossStageItemInfo)

StageResult = _reflection.GeneratedProtocolMessageType('StageResult', (_message.Message,), {
  'DESCRIPTOR' : _STAGERESULT,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.StageResult)
  })
_sym_db.RegisterMessage(StageResult)

MultiStageResult = _reflection.GeneratedProtocolMessageType('MultiStageResult', (_message.Message,), {
  'DESCRIPTOR' : _MULTISTAGERESULT,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.MultiStageResult)
  })
_sym_db.RegisterMessage(MultiStageResult)

CrossStageResult = _reflection.GeneratedProtocolMessageType('CrossStageResult', (_message.Message,), {
  'DESCRIPTOR' : _CROSSSTAGERESULT,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.CrossStageResult)
  })
_sym_db.RegisterMessage(CrossStageResult)

CheckTaskConfigRequest = _reflection.GeneratedProtocolMessageType('CheckTaskConfigRequest', (_message.Message,), {
  'DESCRIPTOR' : _CHECKTASKCONFIGREQUEST,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.CheckTaskConfigRequest)
  })
_sym_db.RegisterMessage(CheckTaskConfigRequest)

CheckTaskConfigResponse = _reflection.GeneratedProtocolMessageType('CheckTaskConfigResponse', (_message.Message,), {
  'DESCRIPTOR' : _CHECKTASKCONFIGRESPONSE,
  '__module__' : 'checker_pb2'
  # @@protoc_insertion_point(class_scope:checker.CheckTaskConfigResponse)
  })
_sym_db.RegisterMessage(CheckTaskConfigResponse)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DICTPATHINFO._serialized_start=26
  _DICTPATHINFO._serialized_end=53
  _LISTPATHINFO._serialized_start=55
  _LISTPATHINFO._serialized_end=84
  _PATHINFO._serialized_start=86
  _PATHINFO._serialized_end=194
  _ITEMINFO._serialized_start=196
  _ITEMINFO._serialized_end=258
  _CROSSSTAGEPOSITIONINFO._serialized_start=260
  _CROSSSTAGEPOSITIONINFO._serialized_end=338
  _CROSSSTAGEITEMINFO._serialized_start=340
  _CROSSSTAGEITEMINFO._serialized_end=436
  _STAGERESULT._serialized_start=439
  _STAGERESULT._serialized_end=598
  _MULTISTAGERESULT._serialized_start=600
  _MULTISTAGERESULT._serialized_end=679
  _CROSSSTAGERESULT._serialized_start=682
  _CROSSSTAGERESULT._serialized_end=884
  _CHECKTASKCONFIGREQUEST._serialized_start=886
  _CHECKTASKCONFIGREQUEST._serialized_end=963
  _CHECKTASKCONFIGRESPONSE._serialized_start=966
  _CHECKTASKCONFIGRESPONSE._serialized_end=1128
# @@protoc_insertion_point(module_scope)
