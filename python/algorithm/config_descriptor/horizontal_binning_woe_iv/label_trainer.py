from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


horizontal_binning_woe_iv_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "horizontal_binning_woe_iv"
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
    "train_info": {
        "train_params": {
            "binning": {
                "method": OneOf("equal_width").set_default_index(0),
                "bins": Integer(5)
            }
        }
    }
}
