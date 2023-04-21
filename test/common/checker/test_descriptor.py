import os
import json
import importlib
from pathlib import Path

from common.checker.compare import compare


def test_local_operator():
    path = Path(Path(__file__).parent.parent.parent.parent) / 'demo' / 'local'
    
    for operator in os.listdir(path):
        conf_path = path / operator / '1party' / 'config' / 'trainer_config_node-1.json'
        conf = json.load(open(conf_path, 'r'))[0]
        
        module_path = '.'.join(['algorithm.config_descriptor', 'local_' + operator, 'label_trainer']) # , 'local_' + operator + '_rule'])
        module = importlib.import_module(module_path)
        rule = getattr(module, 'local_' + operator + '_rule')
        
        result, itemized_result, rule_passed, rule_checked = compare(conf, rule)

        assert rule_passed == rule_checked
        

def test_vertical_operator():
    path = Path(Path(__file__).parent.parent.parent.parent) / 'demo' / 'vertical'
    
    for operator in os.listdir(path):
        for party_num in os.listdir(path / operator):
            for conf_file in os.listdir(path / operator / party_num / 'config'):
                if 'trainer_config' not in conf_file:
                    continue
                
                if operator in ['linear_regression', 'poisson_regression', 'xgboost_distributed']:
                    continue

                conf_path = path / operator / party_num / 'config' / conf_file
                
                if operator == 'feature_selection':
                    conf = json.load(open(conf_path, 'r'))[-1]
                else:
                    conf = json.load(open(conf_path, 'r'))[0]
                
                if 'node-1' in conf_file:
                    role = 'label_trainer'
                elif 'assist_trainer' in conf_file:
                    role = 'assist_trainer'
                else:
                    role = 'trainer'
                    
                module_path = '.'.join(['algorithm.config_descriptor', 'vertical_' + operator, role])
                module = importlib.import_module(module_path)
                rule = getattr(module, 'vertical_' + operator + '_' + role + '_rule')
                
                result, itemized_result, rule_passed, rule_checked = compare(conf, rule)
                
                print(conf)
                print(rule, "---")
                print(result)
                
                assert rule_passed == rule_checked
                
    # operator = 'xgboost_infer'
    # role = 'label_trainer'
    
    # module_path = '.'.join(['algorithm.config_descriptor', 'vertical_' + operator, role])
    # module = importlib.import_module(module_path)
    
    # rule = getattr(module, 'vertical_' + operator + '_' + role + '_rule')

    # conf = {
    #     "identity": "label_trainer",
    #     "model_info": {
    #         "name": "vertical_xgboost"
    #     },
    #     "inference": True,
    #     "input": {
    #         "testset": [],
    #         "pretrained_model": {
    #             "path": "",
    #             "name": ""
    #         }
    #     },
    #     "output": {
    #         "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
    #         "testset": {
    #             "name": "xgb_prediction_test_[STAGE_ID].csv"
    #         }
    #     },
    #     "train_info": {
    #         "train_params": {
    #             "batch_size_val": 40960
    #         }
    #     }
    # }
    
    # result, itemized_result, rule_passed, rule_checked = compare(conf, rule)
                
    # assert rule_passed == rule_checked

        
