from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_feature_preprocess_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_feature_preprocess"
    },
    "input": {
        "__rule__": [Required("trainset"), Optional("valset")],
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
                        "has_label": Bool(False)
                    }
                ).set_default_index(0)
            ).set_default_not_none()
        ]
    },
    "output": {
        "__rule__": [Required("path", "trainset", "model"), Optional("valset")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("local_feature_preprocess_[STAGE_ID].pt")
        },
        "trainset": {
            "name": String("preprocessed_train_[STAGE_ID].csv")
        },
        "valset": {
            "name": String("preprocessed_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params":
        {
            "missing":
            {
                "missing_values": OneOf(Any(None), [Any(None)]).set_default_index(0),  # list,
                "strategy": OneOf("mean", "median", "constant", "most_frequent").set_default_index(0),
                "fill_value": Any(None),
                "missing_features": {
                    "__rule__": Optional(RepeatableSomeOf(String(""))),
                    String(""):
                    {
                        "missing_values": OneOf(Any(None), [Any(None)]).set_default_index(0),
                        "strategy": OneOf("mean", "median", "constant", "most_frequent").set_default_index(0),
                        "fill_value": Any(None)
                    },
                }
            },
            "outlier":
            {
                "outlier_values": OneOf(Any(None), [Any(None)]).set_default_index(0),
                "outlier_features": {
                    "__rule__": Optional(RepeatableSomeOf(String(""))),
                    String(""):
                    {
                        "outlier_values": OneOf(Any(None), [Any(None)]).set_default_index(0)
                    },
                }
            },
            "onehot":
            {
                "onehot_features": {
                    "__rule__": Optional(RepeatableSomeOf(String(""))),
                    String(""): {}
                }
            }
        }
    }
}