from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_kmeans_trainer_rule = {
    "identity": "trainer",
    "model_info": {
        "name": "vertical_kmeans"
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
        ]
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("vertical_kmeans_[STAGE_ID].pkl")
        },
        "result": {
            "name": String("cluster_result_[STAGE_ID].csv")
        },
        "summary": {
            "name": String("cluster_summary_[STAGE_ID].csv")
        }
    },
    "train_info": {
        
    }
}
