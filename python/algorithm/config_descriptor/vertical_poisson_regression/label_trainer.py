from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_poisson_regression_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "vertical_poisson_regression"
    },
    "input": {
        "__rule__": [Optional("pretrained_model"), Required("trainset", "valset")],
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
        ],
        "pretrained_model": {
            "path": String(""),
            "name": String("")
        }
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("vertical_poisson_regression_[STAGE_ID].model")
        },
        "metric_train": {
            "name": String("pr_metric_train_[STAGE_ID].csv")
        },
        "metric_val": {
            "name": String("pr_metric_val_[STAGE_ID].csv")
        },
        "prediction_train": {
            "name": String("pr_prediction_train_[STAGE_ID].csv")
        },
        "prediction_val": {
            "name": String("pr_prediction_val_[STAGE_ID].csv")
        },
        "feature_importance": {
            "name": String("pr_feature_importance_[STAGE_ID].csv")
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
            "global_epoch": Integer(10),
            "batch_size": Integer(128),
            "encryption": {
                "__rule__": OneOf("ckks", "paillier", "plain").set_default("ckks"),
                "ckks": {
                    "poly_modulus_degree": Integer(8192),
                    "coeff_mod_bit_sizes": [
                        RepeatableSomeOf(Integer()).set_default([60, 40, 40, 60])
                    ],
                    "global_scale_bit_size": Integer(40)
                },
                "paillier": {
                    "key_bit_size": OneOf(2048, 4096, 8192).set_default_index(0),
                    "precision": Optional(Integer(7).ge(1)).set_default_not_none(),
                    "djn_on": Bool(True),
                    "parallelize_on": Bool(True)
                },
                "plain": {}
            },
            "optimizer": {
                "lr": Float(0.01),
                "p": OneOf(0, 1, 2).set_default(2),
                "alpha": Float(1e-4)
            },
            "metric": {
                "mse": {},
                "mape": {},
                "mae": {},
                "rmse": {}
            },
            "early_stopping": {
                "key": OneOf("mse", "mape", "mae", "rmse", "loss").set_default_index(-1),
                "patience": Integer(10),
                "delta": Float(0.001)
            },
            "random_seed": Optional(Integer(50))
        }
    }
}
