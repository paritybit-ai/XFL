# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: linear_model.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='linear_model.proto',
  package='model',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12linear_model.proto\x12\x05model\")\n\tStateDict\x12\x0e\n\x06weight\x18\x01 \x03(\x01\x12\x0c\n\x04\x62ias\x18\x02 \x01(\x01\"N\n\x0bLinearModel\x12$\n\nstate_dict\x18\x01 \x01(\x0b\x32\x10.model.StateDict\x12\x19\n\x11suggest_threshold\x18\x02 \x01(\x01\x62\x06proto3'
)




_STATEDICT = _descriptor.Descriptor(
  name='StateDict',
  full_name='model.StateDict',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='weight', full_name='model.StateDict.weight', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bias', full_name='model.StateDict.bias', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=70,
)


_LINEARMODEL = _descriptor.Descriptor(
  name='LinearModel',
  full_name='model.LinearModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state_dict', full_name='model.LinearModel.state_dict', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='suggest_threshold', full_name='model.LinearModel.suggest_threshold', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=150,
)

_LINEARMODEL.fields_by_name['state_dict'].message_type = _STATEDICT
DESCRIPTOR.message_types_by_name['StateDict'] = _STATEDICT
DESCRIPTOR.message_types_by_name['LinearModel'] = _LINEARMODEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StateDict = _reflection.GeneratedProtocolMessageType('StateDict', (_message.Message,), {
  'DESCRIPTOR' : _STATEDICT,
  '__module__' : 'linear_model_pb2'
  # @@protoc_insertion_point(class_scope:model.StateDict)
  })
_sym_db.RegisterMessage(StateDict)

LinearModel = _reflection.GeneratedProtocolMessageType('LinearModel', (_message.Message,), {
  'DESCRIPTOR' : _LINEARMODEL,
  '__module__' : 'linear_model_pb2'
  # @@protoc_insertion_point(class_scope:model.LinearModel)
  })
_sym_db.RegisterMessage(LinearModel)


# @@protoc_insertion_point(module_scope)