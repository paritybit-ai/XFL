from common.utils.fed_conf_parser import FedConfParser


conf = {
    "fed_info": {
       "scheduler": {
            "scheduler": "localhost:55001"
        },
        "trainer": {
            "node-1": "localhost:56001",
            "node-2": "localhost:56002"
        },
        "assist_trainer": {
            "assist_trainer": "localhost:55002"
        }
    },
    "redis_server": "localhost:6379",
    "grpc": {
        "use_tls": False
    } 
}

def test_fed_config_parser():
    res = FedConfParser.parse_dict_conf(conf, 'node-1')
    
    assert res == {'node_id': 'node-1', 
                   'scheduler': {'node_id': 'scheduler', 'host': 'localhost', 'port': '55001', 'use_tls': False}, 
                   'trainer': {'assist_trainer': {'node_id': 'assist_trainer', 'host': 'localhost', 'port': '55002', 'use_tls': False}, 'node-1': {'host': 'localhost', 'port': '56001', 'use_tls': False}, 'node-2': {'host': 'localhost', 'port': '56002', 'use_tls': False}}, 
                   'redis_server': {'host': 'localhost', 'port': '6379'}} 
    
    conf["node_id"] = 'node-2'
    res = FedConfParser.parse_dict_conf(conf, 'node-1')
    assert res == {'node_id': 'node-2', 
                   'scheduler': {'node_id': 'scheduler', 'host': 'localhost', 'port': '55001', 'use_tls': False}, 
                   'trainer': {'assist_trainer': {'node_id': 'assist_trainer', 'host': 'localhost', 'port': '55002', 'use_tls': False}, 'node-1': {'host': 'localhost', 'port': '56001', 'use_tls': False}, 'node-2': {'host': 'localhost', 'port': '56002', 'use_tls': False}}, 
                   'redis_server': {'host': 'localhost', 'port': '6379'}}
    
    del conf["grpc"]
    res = FedConfParser.parse_dict_conf(conf, 'node-1')
    assert res == {'node_id': 'node-2', 
                   'scheduler': {'node_id': 'scheduler', 'host': 'localhost', 'port': '55001', 'use_tls': False}, 
                   'trainer': {'assist_trainer': {'node_id': 'assist_trainer', 'host': 'localhost', 'port': '55002', 'use_tls': False}, 'node-1': {'host': 'localhost', 'port': '56001', 'use_tls': False}, 'node-2': {'host': 'localhost', 'port': '56002', 'use_tls': False}}, 
                   'redis_server': {'host': 'localhost', 'port': '6379'}}