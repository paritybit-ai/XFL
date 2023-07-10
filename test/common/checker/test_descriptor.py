import importlib
import json
import os
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
                print(result)
                
                assert rule_passed == rule_checked
                
                
def test_horizontal_operator():
    path = Path(Path(__file__).parent.parent.parent.parent) / 'demo' / 'horizontal'
    
    for operator in os.listdir(path):
        if operator not in ['logistic_regression', "poisson_regression", "kmeans", "linear_regression", "binning_woe_iv"]:
            continue
                
        for party_num in os.listdir(path / operator):
            if not os.path.isdir(path / operator / party_num):
                continue
            
            for conf_file in os.listdir(path / operator / party_num / 'config'):
                if 'trainer_config' not in conf_file:
                    continue
                
                conf_path = path / operator / party_num / 'config' / conf_file
                print(conf_path, "AAAAAA")
                
                conf = json.load(open(conf_path, 'r'))[0]
                
                if 'assist_trainer' in conf_file:
                    role = 'assist_trainer'
                else:
                    role = 'label_trainer'
                    
                module_path = '.'.join(['algorithm.config_descriptor', 'horizontal_' + operator, role])
                module = importlib.import_module(module_path)
                rule = getattr(module, 'horizontal_' + operator + '_' + role + '_rule')
                
                result, itemized_result, rule_passed, rule_checked = compare(conf, rule)
                print(itemized_result)
                print(conf, "----")

                assert rule_passed == rule_checked
                
                