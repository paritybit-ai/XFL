from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_logistic_regression_trainer_rule = {
    "identity": "trainer",
    "model_info": {
        "name": "vertical_logistic_regression"
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
                    "has_label": Bool(False)
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
                    "has_label": Bool(False)
                }
            ).set_default_indices(0)
        ],
        "pretrained_model": {
            "path": String(""),
            "name": String("")
        }
    },
    "output": {
        "__rule__": [SomeOf("model", "onnx_model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("vertical_logitstic_regression_[STAGE_ID].model")
        },
        "onnx_model": {
            "name": String("vertical_logitstic_regression_[STAGE_ID].onnx")
        },
    },
    "train_info": {
        
    }
}
 