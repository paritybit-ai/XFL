from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_scorecard_fintech_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_scorecard_fintech"
    },
    "input": {
        "trainset": [
            {
                "path": String(""),
                "name": String("")
            }
        ],
        "valset": [
            {
                "path": String(""),
                "name": String("")
            }
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "trainset": {
            "name": String("scorecard_train_[STAGE_ID].csv")
        },
        "valset": {
            "name": String("scorecard_val_[STAGE_ID].csv")
        },
        "decision_table_train": {
            "name": String("scorecard_decision_table_train_[STAGE_ID].csv")
        },
        "decision_table_val": {
            "name": String("scorecard_decision_table_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "metric": {
                "decision_table": {
                    "method": OneOf("equal_frequency", "equal_width").set_default_index(0),
                    "bins": Integer(10),
                    "type": OneOf("score_card")
                }
            },
            "A": Float(500),
            "B": Float(20)
        }
    }
}
