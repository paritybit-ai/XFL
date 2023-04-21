from redis.client import StrictRedis

from common.storage.redis.redis_conn import RedisConn


class TestRedisConn:
    def test_redis_conn(self, mocker):
        rc = RedisConn()
        mocker.patch.object(
            RedisConn, "init_job_id"
        )
        rc.init_redis()

    def test_redis_func(self, mocker):
        rc = RedisConn()

        d = dict()

        def mock_set(key, value, *args, **kwargs):
            d[key] = value

        def mock_get(key, *args, **kwargs):
            return d.get(key)

        def mock_incr(key, *args, **kwargs):
            d[key] = int(d.get(key, 0)) + 1

        def mock_del(key, *args, **kwargs):
            del d[key]

        def mock_exists(key, *args, **kwargs):
            return key in d

        mocker.patch.object(
            StrictRedis, "set", side_effect=mock_set
        )
        mocker.patch.object(
            StrictRedis, "get", side_effect=mock_get
        )
        mocker.patch.object(
            StrictRedis, "incr", side_effect=mock_incr
        )
        mocker.patch.object(
            StrictRedis, "delete", side_effect=mock_del
        )
        mocker.patch.object(
            StrictRedis, "exists", side_effect=mock_exists
        )

        rc.cut_if_exist("what?")
        rc.init_job_id()
        rc.cut_if_exist("XFL_JOB_ID")

        rc.set("XFL_JOB_ID", "1", ex=1)
        rc.put("XFL_JOB_ID", "100")

        assert rc.get("XFL_JOB_ID") == "100"
        rc.set("XFL_JOB_ID", "1")
        assert rc.get("XFL_JOB_ID") == "1"

        rc.incr("XFL_JOB_ID")
        rc.cut("XFL_JOB_ID")
