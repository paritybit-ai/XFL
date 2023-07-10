from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from common.utils.auto_descriptor.torch.optimizer import optimizer
from common.utils.auto_descriptor.torch.lr_scheduler import lr_scheduler
from common.utils.auto_descriptor.torch.lossfunc import lossfunc
from common.utils.auto_descriptor.torch.metrics import metrics
from common.utils.utils import update_dict
from algorithm.core.metrics import metric_dict


horizontal_kmeans_assist_trainer_rule = {
    "identity": "assist_trainer",
    "model_info": {
        "name": "horizontal_kmeans",
        "config": {
            "input_dim": Integer(),
            "num_clusters": Integer(3)
        }
    },
    "input": {
        "__rule__": [Optional("pretrain_model"), Required("valset")],
        "valset": [
            OneOf(
                {
                    "type": "csv",
                    "path": String(),
                    "name": String(),
                    "has_label": Bool(True),
                    "has_id": Bool(False)
                }
            ).set_default_index(0)
        ],
        "pretrain_model": {}
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "model": {
            "name": String("horizontal_kmeans_[STAGE_ID].model")
        },
        "metric_val": {
            "name": String("kmeans_metric_val_[STAGE_ID].csv")
        }
    },
    "train_info": {
        "train_params": {
            "global_epoch": Integer(20),
            "aggregation": {
                "method": {
                    "__rule__": OneOf("fedavg", "fedprox", "scaffold").set_default_index(0),
                    "fedavg": {},
                    "fedprox": {
                        "mu": Float(0.1)
                    },
                    "scaffold": {}
                }
            },
            "encryption": {
                "__rule__": OneOf("otp", "plain").set_default("otp"),
                "otp": {
                    "key_bitlength": OneOf(64, 128).set_default(64),
                    "data_type": "torch.Tensor",
                    "key_exchange": {
                        "key_bitlength": OneOf(3072, 4096, 6144, 8192),
                        "optimized": Bool(True)
                    },
                    "csprng": {
                        "name": OneOf("hmac_drbg").set_default("hmac_drbg"),
                        "method": OneOf("sha1", "sha224", "sha256", "sha384", "sha512").set_default("sha256")
                    }
                },
                "plain": {}
            }
        }
    }
}
