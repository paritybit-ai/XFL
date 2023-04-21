from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


local_data_split_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "local_data_split"
    },
    "input": {
        "dataset":
        [
            OneOf(
                {
                    "type": "csv",
                    "path": String(""),
                    "name": String(""),
                    "has_header": Bool(True),
                    "has_label": Bool(True)
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "trainset":
        {
            "name": String("splitted_train_[STAGE_ID].csv")
        },
        "valset":
        {
            "name": String("splitted_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params":
        {
            "shuffle": Bool(True),
            "max_num_cores": Integer(999),
            "batch_size": Integer(100000),
            "train_weight": Integer(8),
            "val_weight": Integer(2)
        }
    }
}
