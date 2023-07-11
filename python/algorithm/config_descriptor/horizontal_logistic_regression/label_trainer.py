from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


horizontal_logistic_regression_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "horizontal_logistic_regression"
    },
    "input": {
        "trainset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(),
                    "name": String(),
                    "has_label": Bool(True),
                    "has_id": Bool(True)
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "__rule__": [Optional("model"), Optional("onnx_model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("horizontal_logitstic_regression_[STAGE_ID].model")
        },
        "onnx_model": {
            "name": String("horizontal_logitstic_regression_[STAGE_ID].onnx")
        },
        "metric_train": {
            "name": String("lr_metric_train_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "device": OneOf("cpu", "cuda:0"),
        "train_params": {
            "local_epoch": Integer(1),
            "train_batch_size": Integer(64),
        }
    }
}