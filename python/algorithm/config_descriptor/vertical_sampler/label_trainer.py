from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional

vertical_sampler_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "vertical_sampler"
    },
    "input": {
        "dataset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_id": Bool(True),
                    "has_label": Bool(True)
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "sample_id": {
            "name": String("sampled_id_[STAGE_ID].json")
        },
        "dataset": {
            "name": String("sampled_data_[STAGE_ID].pkl")
        }
    },
    "train_info": {
        "train_params": {
            "__rule__": [Optional("marketing_specified"), Required("method", "strategy", "random_seed", "fraction")],
            "method": OneOf("random", "stratify").set_default_index(0),
            "strategy": OneOf("downsample", "upsample").set_default_index(0),
            "random_seed": int(42),
            "fraction": {
                "__rule__": OneOf("number", "percentage", "labeled_percentage").set_default_index(1),
                "number": Integer(),
                "percentage": Float(0.4),
                "labeled_percentage": [RepeatableSomeOf([Integer(), Float()])]
            },
            "marketing_specified": {
                "threshold_method": OneOf("number", "score", "percentage").set_default_index(2),
                "threshold": OneOf(Integer(), Float(0.4)).set_default_index(1)
            }
        }
    }
}
