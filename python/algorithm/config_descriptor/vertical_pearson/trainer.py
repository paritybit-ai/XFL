from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional

vertical_pearson_trainer_rule = {
    "identity": "trainer",
    "model_info": {
        "name": "vertical_pearson"
    },
    "input": {
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
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "corr": {
            "name": String("vertical_pearson_[STAGE_ID].pkl")
        }
    },
    "train_info": {
    }
}
