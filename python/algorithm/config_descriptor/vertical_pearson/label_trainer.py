from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_pearson_label_trainer_rule = {
    "identity": "label_trainer",
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
                    "has_label": Bool(True)
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
        "train_params": {
            "col_index": Integer(-1),
            "col_names": String(""),
            "encryption": {
                "paillier": {
                    "key_bit_size": OneOf(2048, 4096, 8192).set_default_index(0),
                    "precision": Optional(Integer(7)).set_default_not_none(),
                    "djn_on": Bool(True),
                    "parallelize_on": Bool(True)
                },
            },
            "max_num_cores": Integer(999),
            "sample_size": Integer(9999)
        }
    }
}
