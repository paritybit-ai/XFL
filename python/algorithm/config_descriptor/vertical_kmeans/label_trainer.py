from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_kmeans_label_trainer_rule = {
    "identity": "label_trainer",
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
                    "has_label": Bool(True)
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
        "__rule__": Optional("train_params"),
        "train_params": {
            "init": OneOf("random", "kmeans++").set_default("random"),
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
            },
            "k": Integer(5),
            "max_iter": Integer(50),
            "tol": Float(1e-6),
            "random_seed": Float(50)
        }
    }
}
 