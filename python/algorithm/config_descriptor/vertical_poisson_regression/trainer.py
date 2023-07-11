from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_poisson_regression_trainer_rule = {
    "identity": "trainer",
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
        }
    },
    "train_info": {
    }
}
