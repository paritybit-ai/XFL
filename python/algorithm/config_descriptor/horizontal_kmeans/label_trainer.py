from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


horizontal_kmeans_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "horizontal_kmeans"
    },
    "input": {
        "trainset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(),
                    "name": String(),
                    "has_label": Bool(True),
                    "has_id": Bool(False)
                }
            ).set_default_index(0)
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "metric_train": {
            "name": String("kmeans_metric_train_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "local_epoch": Integer(1)
        }
    }
}
