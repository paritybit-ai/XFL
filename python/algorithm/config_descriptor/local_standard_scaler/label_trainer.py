from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_standard_scaler_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_standard_scaler"
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
            Optional(OneOf(
                    {
                        "type": "csv",
                        "path": String(""),
                        "name": String(""),
                        "has_id": Bool(True),
                        "has_label": Bool(True)
                    }
                ).set_default_index(0)
            ).set_default_not_none()
        ]
    },
    "output": {
        "__rule__": [SomeOf("model", "proto_model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("local_standard_scaler_[STAGE_ID].model")
        },
        "proto_model": {
            "name": String("local_standard_scaler_[STAGE_ID].pmodel")
        },
        "trainset": {
            "name": String("standardized_train_[STAGE_ID].csv")
        },
        "valset": {
            "name": String("standardized_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "with_mean": Bool(True),
            "with_std": Bool(True),
            "feature_standard": {
                "__rule__": Optional(String()),
                String(): {
                    "with_mean": Bool(False),
                    "with_std": Bool(False)
                }
            }
        }
    }
}