from unittest.mock import patch

import redis

from service.fed_node import FedNode
from common.storage.redis.redis_conn import RedisConn


# def test_RedisConn(mocker):
#     patch("redis.StrictRedis", "rs")
#     # patch("FedNode.redis_host", "localhost")
#     # patch("FedNode.redis_port", 6379)
    
#     mocker.patch.object(FedNode, 'redis_host', return_value="localhost")
#     mocker.patch.object(FedNode, 'redis_port', return_value=6379)
#     mocker.patch.object(redis, "ConnectionPool", side_effect=lambda x: "pool")
#     mocker.patch.object(redis, "StrictRedis", side_effect=lambda x: "strict_redis")
#     conn = RedisConn.init_redis()
    