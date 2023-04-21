from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_normalization_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_normalization"
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
            "name": String("local_normalization_[STAGE_ID].model")
        },
        "proto_model": {
            "name": String("local_normalization_[STAGE_ID].pmodel")
        },
        "trainset": {
            "name": String("normalized_train_[STAGE_ID].csv")
        },
        "valset": {
            "name": String("normalized_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "__rule__": [Optional("feature_norm"), Required("norm", "axis")],
            "norm": OneOf("l1", "l2", "max").set_default_index(0),
            "axis": OneOf(0, 1).set_default(0),
            "feature_norm": {
                "__rule__": Optional(RepeatableSomeOf(String(""))),
                String(""): {
                    "norm": OneOf("l1", "l2", "max").set_default_index(0)
                }
            }
        }
    }
}
