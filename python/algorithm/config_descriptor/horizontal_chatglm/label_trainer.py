from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


horizontal_chatglm_label_trainer_rule = {
    "identity": "label_trainer",
    "model_info": {
        "name": "horizontal_chatglm"
    },
    "input": {
        "__rule__": [Optional("trainset"), Optional("adater_model"), Optional("pretrain_model")],
        "trainset": [
            {
                "type": "QA",
                "path": String()
            }
        ],
        "pretrained_model": {
            "path": String()
        },
        "adapter_model": {
            "path": String()
        }
    },
    "output": {
        "path":  String("/opt/checkpoints/[JOB_ID]/[NODE_ID]")
    },
    "train_info": {
        "train_params": {
            "trainer": {
                "per_device_train_batch_size": Integer(1),
                "gradient_accumulation_steps": Integer(4),
                "save_strategy": OneOf("steps", "no"),
                "torch_compile": Bool(False),
                "no_cuda": Bool(False)
            }
        }
    }
}
