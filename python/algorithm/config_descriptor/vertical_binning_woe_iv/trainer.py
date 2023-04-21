from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional

vertical_binning_woe_iv_trainer_rule = {
    "identity": "trainer",
    "model_info": {
        "name": "vertical_binning_woe_iv"
    },
    "input": {
        "trainset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_id": Bool(True),
                    "has_label": Bool(False),
                    "nan_list": [Optional(RepeatableSomeOf(Any()))]
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "split_points": {
            "name": String("binning_split_points_[STAGE_ID].json")
        }
    },
    "train_info": {
        "train_params": {
            "max_num_cores": Integer(2)
        }
    }
}
