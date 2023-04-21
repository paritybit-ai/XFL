from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_data_statistic_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_data_statistic"
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
        "summary": {
            "name": String("data_summary_[STAGE_ID].json")
        }
    },
    "train_info": {
        "train_params": {
            "quantile": [RepeatableSomeOf(Float(0.25))]
        }
    }
}