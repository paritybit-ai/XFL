# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import checker_pb2 as checker__pb2
import commu_pb2 as commu__pb2
import control_pb2 as control__pb2
import scheduler_pb2 as scheduler__pb2
import status_pb2 as status__pb2


class SchedulerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getConfig = channel.unary_unary(
                '/scheduler.Scheduler/getConfig',
                request_serializer=scheduler__pb2.GetConfigRequest.SerializeToString,
                response_deserializer=scheduler__pb2.GetConfigResponse.FromString,
                )
        self.post = channel.stream_unary(
                '/scheduler.Scheduler/post',
                request_serializer=commu__pb2.PostRequest.SerializeToString,
                response_deserializer=commu__pb2.PostResponse.FromString,
                )
        self.control = channel.unary_unary(
                '/scheduler.Scheduler/control',
                request_serializer=control__pb2.ControlRequest.SerializeToString,
                response_deserializer=control__pb2.ControlResponse.FromString,
                )
        self.status = channel.unary_unary(
                '/scheduler.Scheduler/status',
                request_serializer=status__pb2.StatusRequest.SerializeToString,
                response_deserializer=status__pb2.StatusResponse.FromString,
                )
        self.getAlgorithmList = channel.unary_unary(
                '/scheduler.Scheduler/getAlgorithmList',
                request_serializer=scheduler__pb2.GetAlgorithmListRequest.SerializeToString,
                response_deserializer=scheduler__pb2.GetAlgorithmListResponse.FromString,
                )
        self.recProgress = channel.unary_unary(
                '/scheduler.Scheduler/recProgress',
                request_serializer=scheduler__pb2.RecProgressRequest.SerializeToString,
                response_deserializer=scheduler__pb2.RecProgressResponse.FromString,
                )
        self.getStage = channel.unary_unary(
                '/scheduler.Scheduler/getStage',
                request_serializer=scheduler__pb2.GetStageRequest.SerializeToString,
                response_deserializer=scheduler__pb2.GetStageResponse.FromString,
                )
        self.checkTaskConfig = channel.unary_unary(
                '/scheduler.Scheduler/checkTaskConfig',
                request_serializer=checker__pb2.CheckTaskConfigRequest.SerializeToString,
                response_deserializer=checker__pb2.CheckTaskConfigResponse.FromString,
                )


class SchedulerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def getConfig(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def post(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def control(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getAlgorithmList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recProgress(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getStage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def checkTaskConfig(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SchedulerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getConfig': grpc.unary_unary_rpc_method_handler(
                    servicer.getConfig,
                    request_deserializer=scheduler__pb2.GetConfigRequest.FromString,
                    response_serializer=scheduler__pb2.GetConfigResponse.SerializeToString,
            ),
            'post': grpc.stream_unary_rpc_method_handler(
                    servicer.post,
                    request_deserializer=commu__pb2.PostRequest.FromString,
                    response_serializer=commu__pb2.PostResponse.SerializeToString,
            ),
            'control': grpc.unary_unary_rpc_method_handler(
                    servicer.control,
                    request_deserializer=control__pb2.ControlRequest.FromString,
                    response_serializer=control__pb2.ControlResponse.SerializeToString,
            ),
            'status': grpc.unary_unary_rpc_method_handler(
                    servicer.status,
                    request_deserializer=status__pb2.StatusRequest.FromString,
                    response_serializer=status__pb2.StatusResponse.SerializeToString,
            ),
            'getAlgorithmList': grpc.unary_unary_rpc_method_handler(
                    servicer.getAlgorithmList,
                    request_deserializer=scheduler__pb2.GetAlgorithmListRequest.FromString,
                    response_serializer=scheduler__pb2.GetAlgorithmListResponse.SerializeToString,
            ),
            'recProgress': grpc.unary_unary_rpc_method_handler(
                    servicer.recProgress,
                    request_deserializer=scheduler__pb2.RecProgressRequest.FromString,
                    response_serializer=scheduler__pb2.RecProgressResponse.SerializeToString,
            ),
            'getStage': grpc.unary_unary_rpc_method_handler(
                    servicer.getStage,
                    request_deserializer=scheduler__pb2.GetStageRequest.FromString,
                    response_serializer=scheduler__pb2.GetStageResponse.SerializeToString,
            ),
            'checkTaskConfig': grpc.unary_unary_rpc_method_handler(
                    servicer.checkTaskConfig,
                    request_deserializer=checker__pb2.CheckTaskConfigRequest.FromString,
                    response_serializer=checker__pb2.CheckTaskConfigResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'scheduler.Scheduler', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Scheduler(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def getConfig(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/getConfig',
            scheduler__pb2.GetConfigRequest.SerializeToString,
            scheduler__pb2.GetConfigResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def post(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/scheduler.Scheduler/post',
            commu__pb2.PostRequest.SerializeToString,
            commu__pb2.PostResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def control(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/control',
            control__pb2.ControlRequest.SerializeToString,
            control__pb2.ControlResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/status',
            status__pb2.StatusRequest.SerializeToString,
            status__pb2.StatusResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getAlgorithmList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/getAlgorithmList',
            scheduler__pb2.GetAlgorithmListRequest.SerializeToString,
            scheduler__pb2.GetAlgorithmListResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recProgress(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/recProgress',
            scheduler__pb2.RecProgressRequest.SerializeToString,
            scheduler__pb2.RecProgressResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getStage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/getStage',
            scheduler__pb2.GetStageRequest.SerializeToString,
            scheduler__pb2.GetStageResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def checkTaskConfig(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/scheduler.Scheduler/checkTaskConfig',
            checker__pb2.CheckTaskConfigRequest.SerializeToString,
            checker__pb2.CheckTaskConfigResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
