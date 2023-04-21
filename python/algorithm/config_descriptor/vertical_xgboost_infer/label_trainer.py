from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_xgboost_infer_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "vertical_xgboost"
    },
    "inference": True,
    "input": {
        "testset": [
            Optional(
                OneOf(
                    {
                        "type": "csv",
                        "path": String(""),
                        "name": String(""),
                        "has_id": Bool(True),
                        "has_label": Bool(True)
                    }
                ).set_default_index(0)
            )
        ],
        "pretrained_model": {
            "path": String(""),
            "name": String("")
        }
    },
    "output": {
        "__rule__": [Optional("path"), Optional("testset")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "testset": {
            "name": String("xgb_prediction_test_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "batch_size_val": Integer(40960)
        }
    }
}
