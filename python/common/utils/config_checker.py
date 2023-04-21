import os
import importlib
import traceback
from collections import Counter

from common.checker.compare import compare
from common.utils.logger import logger
from common.utils.config_parser import replace_variable


def find_rule_class(fed_type, operator_name, role, inference):
    try:
        if inference:
            operator_name += '_infer'
        module_path = '.'.join(['algorithm.config_descriptor', fed_type + '_' + operator_name, role]) # , 'local_' + operator + '_rule'])
        module = importlib.import_module(module_path)
    except Exception: # ModuleNotFoundError:
        logger.warning(traceback.format_exc())
        return None
    
    try:
        if fed_type == 'local':
            rule = getattr(module, fed_type + '_' + operator_name + '_rule')
        elif fed_type == 'vertical':
            rule = getattr(module, fed_type + '_' + operator_name + '_' + role + '_rule')
        else:
            return None
            
    except Exception:
        return None
    
    return rule


def check_stage_train_conf(conf):
    role = conf.get("identity")
    name = conf.get('model_info', {}).get('name')
    
    # name = 'vertical_logistic_regression'
    
    fed_type = name.split('_')[0]
    operator_name = '_'.join(name.split('_')[1:])
    inference = True if conf.get('inference') else False
    
    res = {
            "result": {},
            "summary": (0, 0),
            "message": 'Rule not found.'
        }
    
    if not role or not name:
        res["message"] = f"Role {role} or Name {name} not valid."
        return res
    
    rule = find_rule_class(fed_type, operator_name, role, inference)
    if not rule:
        return res
    
    try:
        result, itemized_result, rule_passed, rule_checked = compare(conf, rule)
    except Exception:
        logger.warning(traceback.format_exc())
        logger.info("Error when checking train_config.")
        return res
    
    res = {
        "result": result,
        "itemized_result": itemized_result,
        "summary": (rule_passed, rule_checked),
        "message": 'Config checked.'
    }
    return res


def check_multi_stage_train_conf(conf: list):
    if not isinstance(conf, list):
        return [], [(0, 1)], "Not a list"
    
    res = {
        "result": [],
        "itemized_result": [],
        "summary": [],
        "message": []
    }
    
    for stage_conf in conf:
        if not isinstance(stage_conf, dict):
            stage_result = {"rule_passed": 0, "rule_checked": 1}
            stage_message = "Not a dict."
        else:
            report = check_stage_train_conf(stage_conf)
            stage_result = report["result"]
            stage_itemized_result = report["itemized_result"]
            stage_summary = report["summary"]
            stage_message = report["message"]
            
        res["result"].append(stage_result)
        res["itemized_result"].append(stage_itemized_result)
        res["summary"].append(stage_summary)
        res["message"].append(stage_message)

    return res


def check_cross_stage_input_output(conf: list, ignore_list: list = []):
    input_dict = {}  
    output_dict = {}
    """
    {
        0: [
            {
                "key_chain": ["input", "trainset"], 
                "value": "/opt/dataset/a.csv"
            }
        ]
    }
    
    """

    for stage_id, stage_conf in enumerate(conf):
        input = stage_conf.get("input", {})
        path = input.get("path", "")
        input_path = []
        for key in input:
            if isinstance(input[key], list):
                for item in input[key]:
                    local_path = item.get("path", "") or path
                    local_name = item.get("name", "")
                    if isinstance(local_name, list):
                        for name in local_name:
                            input_path.append(
                                {
                                    "key_chain": ["input", key],
                                    "value": os.path.join(local_path, name)
                                }
                            )
                    else:
                        input_path.append(
                            {
                                "key_chain": ["input", key],
                                "value": os.path.join(local_path, local_name)
                            }
                        )
            elif isinstance(input[key], dict):
                item = input[key]
                local_path = item.get("path", "") or path
                local_name = item.get("name", "")
                
                if isinstance(local_name, list):
                    for name in local_name:
                        input_path.append(
                            {
                                "key_chain": ["input", key],
                                "value": os.path.join(local_path, name)
                            }
                        )
                else:
                    input_path.append(
                        {
                            "key_chain": ["input", key],
                            "value": os.path.join(local_path, local_name)
                        }
                    )
                
                    
        input_dict[stage_id] = input_path
        
        output = stage_conf.get("output", {})
        path = output.get("path", "")
        output_path = []
        for key in output:
            if isinstance(output[key], dict):
                local_path = output[key].get("path") or path
                local_name = output[key].get("name", "")
                output_path.append(
                    {
                        "key_chain": ["output", key],
                        "value": os.path.join(local_path, local_name)
                    }
                )
                
        output_dict[stage_id] = output_path
        
    input_dict_a = {k: replace_variable(v, stage_id=k, job_id='JOB_ID', node_id='NODE_ID') for k, v in input_dict.items()}
    output_dict_a = {k: replace_variable(v, stage_id=k, job_id='JOB_ID', node_id='NODE_ID') for k, v in output_dict.items()}
    
    def find_duplicated_and_blank(in_dict, duplicated=True):
        result = {
            "duplicated": [],
            "blank": [],
            "nonexistent": []
        }
        
        stage_id_list = []
        key_chain_list = []
        value_list = []

        for stage_id in in_dict:
            for path_dict in in_dict[stage_id]:
                stage_id_list.append(stage_id)
                key_chain_list.append(path_dict['key_chain'])
                value_list.append(path_dict['value'])
                
        if duplicated:
            count_result = dict(Counter(value_list))
            for k in count_result:
                # find duplicated
                if count_result[k] > 1:
                    index = [i for i, v in enumerate(value_list) if v == k]
                    if index:
                        result['duplicated'].append(
                            {
                                "value": k,
                                "position": [
                                    {
                                        "stage": stage_id_list[i],
                                        "key_chain": key_chain_list[i],
                                    } for i in index
                                ]
                            }
                        )
            
        # find blank
        index = [i for i, v in enumerate(value_list) if v.strip() == '']
        if index:
            result['blank'].append(
                {
                    "value": '',
                    "position": [
                        {
                            "stage": stage_id_list[i],
                            "key_chain": key_chain_list[i],
                        } for i in index
                    ]
                }
            )
        return result
            
    def find_nonexistent(input_dict, output_dict, ignore_list):
        result = {
            "duplicated": [],
            "blank": [],
            "nonexistent": []
        }
        stage_id_list = []
        key_chain_list = []
        value_list = []
        
        for stage_id in input_dict:
            for path_dict in input_dict[stage_id]:
                stage_id_list.append(stage_id)
                key_chain_list.append(path_dict['key_chain'])
                value_list.append(path_dict['value'])
                
        output_stage_id_list = []
        output_key_chain_list = []
        output_value_list = []
        
        for stage_id in output_dict:
            for path_dict in output_dict[stage_id]:
                output_stage_id_list.append(stage_id)
                output_key_chain_list.append(path_dict['key_chain'])
                output_value_list.append(path_dict['value'])
                
        for i, stage_id in enumerate(stage_id_list):
            ids = [j for j, stage in enumerate(output_stage_id_list) if stage < stage_id]
            if value_list[i] not in [output_value_list[j] for j in ids] and value_list[i] not in ignore_list:
                result['nonexistent'].append(
                    {
                        "value": value_list[i],
                        "position": [
                            {
                                "stage": stage_id_list[i],
                                "key_chain": key_chain_list[i],
                            }
                        ]

                    }
                )
        return result
    
    result = {
        "duplicated": [],
        "blank": [],
        "nonexistent": []
    }
    
    r1 = find_duplicated_and_blank(input_dict_a, duplicated=False)
    r2 = find_duplicated_and_blank(output_dict_a)
    r3 = find_nonexistent(input_dict_a, output_dict_a, ignore_list)
    
    result["duplicated"] += r1["duplicated"]
    result["duplicated"] += r2["duplicated"]
    result["blank"] += r1["blank"]
    result["blank"] += r2["blank"]
    result["nonexistent"] += r3["nonexistent"]
    return result


if __name__ == "__main__":
    # path = '/mnt/c/Documents and Settings/wanghong/workspace/federated-learning/demo/vertical/xgboost/2party_env/config/trainer_config_node-1.json'
    # import json
    # conf = json.load(open(path, 'r'))
    
    conf = \
        [
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_binning_woe_iv_fintech"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True,
                            "nan_list": [
                            ]
                        }
                    ]
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "vertical_binning_woe_iv_[STAGE_ID].json"
                    },
                    "iv": {
                        "name": "woe_iv_result_[STAGE_ID].json"
                    },
                    "split_points": {
                        "name": "binning_split_points_[STAGE_ID].json"
                    },
                    "trainset": {
                        "name": "fintech_woe_map_train_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "interaction_params": {
                        "save_model": True
                    },
                    "train_params": {
                        "encryption": {
                            "paillier": {
                                "key_bit_size": 2048,
                                "precision": 7,
                                "djn_on": True,
                                "parallelize_on": True
                            }
                        },
                        "binning": {
                            "method": "equal_width",
                            "bins": 5
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_feature_selection"
                },
                "input": {
                    "iv_result": {
                        "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "woe_iv_result_[STAGE_ID-1].json"
                    },
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/opt/dataset/testing/fintech",
                            "name": "banking_guest_train_v01_20220216_TL.csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "feature_selection_[STAGE_ID].pkl"
                    },
                    "trainset": {
                        "name": "selected_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "selected_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "filter": {
                            "common": {
                                "metrics": "iv",
                                "filter_method": "threshold",
                                "threshold": 0.01
                            }
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_pearson"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "corr": {
                        "name": "vertical_pearson_[STAGE_ID].pkl"
                    }
                },
                "train_info": {
                    "train_params": {
                        "col_index": -1,
                        "col_names": "",
                        "encryption": {
                            "paillier": {
                                "key_bit_size": 2048,
                                "precision": 6,
                                "djn_on": True,
                                "parallelize_on": True
                            }
                        },
                        "max_num_cores": 999,
                        "sample_size": 9999
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_feature_selection"
                },
                "input": {
                    "corr_result": {
                        "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "vertical_pearson_[STAGE_ID-1].pkl"
                    },
                    "iv_result": {
                        "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                        "name": "woe_iv_result_[STAGE_ID-3].json"
                    },
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-2].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_val_[STAGE_ID-2].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "feature_selection_[STAGE_ID].pkl"
                    },
                    "trainset": {
                        "name": "selected_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "selected_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "filter": {
                            "common": {
                                "metrics": "iv",
                                "filter_method": "threshold",
                                "threshold": 0.01
                            },
                            "correlation": {
                                "sort_metric": "iv",
                                "correlation_threshold": 0.7
                            }
                        }
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "local_normalization"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "selected_val_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ]
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "local_normalization_[STAGE_ID].pt"
                    },
                    "trainset": {
                        "name": "normalized_train_[STAGE_ID].csv"
                    },
                    "valset": {
                        "name": "normalized_val_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "train_params": {
                        "norm": "max",
                        "axis": 0
                    }
                }
            },
            {
                "identity": "label_trainer",
                "model_info": {
                    "name": "vertical_logistic_regression"
                },
                "input": {
                    "trainset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "normalized_train_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "valset": [
                        {
                            "type": "csv",
                            "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                            "name": "normalized_val_[STAGE_ID-1].csv",
                            "has_id": True,
                            "has_label": True
                        }
                    ],
                    "pretrained_model": {
                        "path": "",
                        "name": ""
                    }
                },
                "output": {
                    "path": "/opt/checkpoints/[JOB_ID]/[NODE_ID]",
                    "model": {
                        "name": "vertical_logitstic_regression_[STAGE_ID].pt"
                    },
                    "metric_train": {
                        "name": "lr_metric_train_[STAGE_ID].csv"
                    },
                    "metric_val": {
                        "name": "lr_metric_val_[STAGE_ID].csv"
                    },
                    "prediction_train": {
                        "name": "lr_prediction_train_[STAGE_ID].csv"
                    },
                    "prediction_val": {
                        "name": "lr_prediction_val_[STAGE_ID].csv"
                    },
                    "ks_plot_train": {
                        "name": "lr_ks_plot_train_[STAGE_ID].csv"
                    },
                    "ks_plot_val": {
                        "name": "lr_ks_plot_val_[STAGE_ID].csv"
                    },
                    "decision_table_train": {
                        "name": "lr_decision_table_train_[STAGE_ID].csv"
                    },
                    "decision_table_val": {
                        "name": "lr_decision_table_val_[STAGE_ID].csv"
                    },
                    "feature_importance": {
                        "name": "lr_feature_importance_[STAGE_ID].csv"
                    }
                },
                "train_info": {
                    "interaction_params": {
                        "save_frequency": -1,
                        "write_training_prediction": True,
                        "write_validation_prediction": True,
                        "echo_training_metrics": True
                    },
                    "train_params": {
                        "global_epoch": 2,
                        "batch_size": 512,
                        "encryption": {
                            "ckks": {
                                "poly_modulus_degree": 8192,
                                "coeff_mod_bit_sizes": [
                                    60,
                                    40,
                                    40,
                                    60
                                ],
                                "global_scale_bit_size": 40
                            }
                        },
                        "optimizer": {
                            "lr": 0.01,
                            "p": 2,
                            "alpha": 1e-4
                        },
                        "metric": {
                            "decision_table": {
                                "method": "equal_frequency",
                                "bins": 10
                            },
                            "acc": {},
                            "precision": {},
                            "recall": {},
                            "f1_score": {},
                            "auc": {},
                            "ks": {}
                        },
                        "early_stopping": {
                            "key": "acc",
                            "patience": 10,
                            "delta": 0
                        },
                        "random_seed": 50
                    }
                }
            }
        ]

    result = check_multi_stage_train_conf(conf)
    print(result)
    
    result = check_cross_stage_input_output(conf)
    print(result)
    
    conf = [
        {
            "identity": "label_trainer",
            "model_info": {
                "name": "vertical_xgboost"
            },
            "inference": True
        }
    ]
    
    result = check_multi_stage_train_conf(conf)
    print(result)
    
    
    
