from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional

vertical_sampler_trainer_rule = {
    "identity": "trainer",
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
                    "has_label": Bool(False)
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
            "name": String("sampled_data_[STAGE_ID].csv")
        }
    },
    "train_info": {
    }
}
