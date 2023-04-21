from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_xgboost_trainer_rule = {
    "identity": "trainer",
    "model_info": {
        "name": "vertical_xgboost"
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
        ],
        "valset": [
            RepeatableSomeOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_id": Bool(True),
                    "has_label": Bool(False)
                }
            ).set_default_indices(0)
        ]
    },
    "output": {
        "__rule__": [SomeOf("model", "proto_model")],
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("vertical_xgboost_[STAGE_ID].model")
        },
        "proto_model": {
            "name": String("vertical_xgboost_[STAGE_ID].pmodel")
        }
    },
    "train_info": {
        "train_params": {
            "max_num_cores": Integer(999).ge(1),
            "downsampling": {
                "column": {
                    "rate": Float(1.0).gt(0).le(1)
                }
            },
            "category": {
                "cat_features": {
                    "col_index": String(""),
                    "col_names": [Optional(RepeatableSomeOf(String("")))],
                    "max_num_value": Integer(0).ge(0),
                    "col_index_type": OneOf("inclusive", "exclusive").set_default_index(0),
                    "col_names_type": OneOf("inclusive", "exclusive").set_default_index(0),
                    "max_num_value_type": OneOf("intersection", "union").set_default_index(1)
                }
            },
            "advanced": {
                "row_batch": Integer(40000).ge(1),
                "col_batch": Integer(64).ge(1)
            }
        }
    }
}
