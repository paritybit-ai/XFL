from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_xgboost_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "vertical_xgboost"
    },
    "input": {
        "trainset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_id": Bool(True),
                    "has_label": Bool(True)
                }
            ).set_default_index(0)
        ],
        "valset": [
            RepeatableSomeOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_id": Bool(True),
                    "has_label": Bool(True)
                }
            ).set_default_indices(0)
        ]
    },
    "output": {
        "__rule__": [SomeOf("model", "proto_model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("vertical_xgboost_[STAGE_ID].model")
        },
        "proto_model": {
            "name": String("vertical_xgboost_[STAGE_ID].pmodel")
        },
        "metric_train": {
            "name": String("xgb_metric_train_[STAGE_ID].csv")
        },
        "metric_val": {
            "name": String("xgb_metric_val_[STAGE_ID].csv")
        },
        "prediction_train": {
            "name": String("xgb_prediction_train_[STAGE_ID].csv")
        },
        "prediction_val": {
            "name": String("xgb_prediction_val_[STAGE_ID].csv")
        },
        "ks_plot_train": {
            "name": String("xgb_ks_plot_train_[STAGE_ID].csv")
        },
        "ks_plot_val": {
            "name": String("xgb_ks_plot_val[STAGE_ID].csv")
        },
        "decision_table_train": {
            "name": String("xgb_decision_table_train_[STAGE_ID].csv")
        },
        "decision_table_val": {
            "name": String("xgb_decision_table_val_[STAGE_ID].csv")
        },
        "feature_importance": {
            "name": String("xgb_feature_importance_[STAGE_ID].csv")
        },
        "plot_ks": {
            "name": "xgb_plot_ks_[STAGE_ID].json"
        },
        "plot_roc": {
            "name": "xgb_plot_roc_[STAGE_ID].json"
        },
        "plot_lift": {
            "name": "xgb_plot_lift_[STAGE_ID].json"
        },
        "plot_gain": {
            "name": "xgb_plot_gain_[STAGE_ID].json"
        },
        "plot_precision_recall": {
            "name": "xgb_plot_precision_recall_[STAGE_ID].json"
        },
        "plot_feature_importance": {
            "name": "xgb_plot_feature_importance_[STAGE_ID].json"
        },
        "plot_loss": {
            "name": "xgb_plot_loss_[STAGE_ID].json"
        }
    },
    "train_info": {
        "interaction_params": {
            "save_frequency": Integer(-1).ge(-1),
            "echo_training_metrics": Bool(True),
            "write_training_prediction": Bool(True),
            "write_validation_prediction": Bool(True)
        },
        "train_params": {
            "lossfunc": {
                "__rule__": OneOf("BCEWithLogitsLoss").set_default_index(0),
                "BCEWithLogitsLoss": {}
            },
            "num_trees": Integer(30).ge(1),
            "learning_rate": Float(0.3).gt(0),
            "gamma": Float(0),
            "lambda_": Float(1.0),
            "max_depth": Integer(3).ge(1),
            "num_bins": Integer(16).ge(2).le(65535),
            "min_split_gain": Float(0).ge(0),
            "min_sample_split": Integer(20).ge(1),
            "feature_importance_type": OneOf("gain", "split").set_default_index(0),
            "max_num_cores": Integer(999).ge(1),
            "batch_size_val": Integer(40960).ge(1),
            "downsampling": {
                "column": {
                    "rate": Float(1.0).gt(0).le(1)
                },
                "row": {
                    "run_goss": Bool(True),
                    "top_rate": Float(0.4).gt(0).le(1),
                    "other_rate": Float(0.4).gt(0).le(1).add_rule(lambda x, y: x + y["train_info"]["train_params"]["downsampling"]["row"]["top_rate"] <= 1, "top_rate + other_rate <=1")
                }
            },
            "category": {
                "cat_smooth": Float(1.0),
                "cat_features": {
                    "col_index": String(""),
                    "col_names": [Optional(RepeatableSomeOf(String("")))],
                    "max_num_value": Integer(0).ge(0),
                    "col_index_type": OneOf("inclusive", "exclusive").set_default_index(0),
                    "col_names_type": OneOf("inclusive", "exclusive").set_default_index(0),
                    "max_num_value_type": OneOf("intersection", "union").set_default_index(1)
                }
            },
            "metric": {
                "__rule__": [Optional("decision_table"), Required("acc", "precision", "recall", "f1_score", "auc", "ks")],

                "acc": {},
                "precision": {},
                "recall": {},
                "f1_score": {},
                "auc": {},
                "ks": {},
                "decision_table": {
                    "method": OneOf("equal_frequency", "equal_width").set_default_index(0),
                    "bins": Integer(10).ge(2)
                }
            },
            "early_stopping": {
                # 这里的key必须是在metric里配置过的key
                "key": OneOf("acc", "precision", "recall", "f1_score", "auc", "ks").set_default_index(-1).add_rule(lambda x, y: x in y["train_info"]["train_params"]["metric"].keys(), "should in metric"),
                "patience": Integer(10).ge(-1),
                "delta": Float(0.001).gt(0)
            },
            "encryption": {
                "__rule__": OneOf("paillier", "plain").set_default_index(0),

                "paillier": {
                    "key_bit_size": OneOf(2048, 4096, 8192).set_default_index(0),
                    "precision": Optional(Integer(7).ge(1)).set_default_not_none(),
                    "djn_on": Bool(True),
                    "parallelize_on": Bool(True)
                },
                "plain": {}
            }
        }
    }
}
