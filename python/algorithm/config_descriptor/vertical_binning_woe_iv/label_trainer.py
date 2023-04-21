from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional

vertical_binning_woe_iv_label_trainer_rule = {
    "identity": "label_trainer",
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
                    "has_label": Bool(True),
                    "nan_list": [Optional(RepeatableSomeOf(Any()))]
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "iv": {
            "name": String("woe_iv_result_[STAGE_ID].json")
        },
        "split_points": {
            "name": String("binning_split_points_[STAGE_ID].json")
        }
    },
    "train_info": {
        "train_params": {
            "encryption": {
                "__rule__": OneOf("paillier", "plain").set_default_index(0),

                "paillier": {
                    "key_bit_size": OneOf(2048, 4096, 8192).set_default_index(0),
                    "precision": Optional(Integer(7)).set_default_not_none(),
                    "djn_on": Bool(True),
                    "parallelize_on": Bool(True)
                },
                "plain": {}
            },
            "binning": {
                "method": OneOf("equal_frequency", "equal_width").set_default_index(1),
                "bins": Integer(5)
            }
        }
    }
}
