from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_feature_selection_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "vertical_feature_selection"
    },
    "input": {
        "__rule__": [Optional("corr_result", "valset"), Required("trainset", "iv_result")],
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
        "iv_result": {
            "path": String(""),
            "name": String("")
        },
        "corr_result": {
            "path": String(""),
            "name": String("")
        }
    },
    "output": {
        "__rule__": [Optional("valset"), Required("path", "trainset", "model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "trainset": {
            "name": String("selected_train_[STAGE_ID].csv")
        },
        "valset": {
            "name": String("selected_val_[STAGE_ID].csv")
        },
        "model": {
            "name": String("vertical_feature_selection_[STAGE_ID].pkl")
        }
    },
    "train_info": {
        "train_params": {
            "filter": {
                "__rule__": [Optional("correlation").add_rule(lambda x, y: "corr_result" in y["input"].keys()), Required("common")],
                "common": {
                    "metrics": "iv",
                    "filter_method": "threshold",
                    "threshold": Float(0.01)
                },
                "correlation": {
                    "sort_metric": "iv",
                    "correlation_threshold": Float(0.7)
                }
            }
        }
    }
}
