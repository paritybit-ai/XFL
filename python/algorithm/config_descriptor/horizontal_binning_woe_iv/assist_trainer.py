from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from common.utils.auto_descriptor.torch.optimizer import optimizer
from common.utils.auto_descriptor.torch.lr_scheduler import lr_scheduler
from common.utils.auto_descriptor.torch.lossfunc import lossfunc
from common.utils.auto_descriptor.torch.metrics import metrics
from common.utils.utils import update_dict
from algorithm.core.metrics import metric_dict


horizontal_binning_woe_iv_assist_trainer_rule = {
    "identity": "assist_trainer",
    "model_info": {
        "name": "horizontal_binning_woe_iv"
    },
    "output": {
        "path": String("/opt/checkpoints/[JOB_ID]/[NODE_ID]"),
        "result": {
            "name": String("woe_iv_result_[STAGE_ID].json")
        }
    },
    "train_info": {
        "train_params": {
            "encryption": {
                "__rule__": OneOf("otp", "plain").set_default("otp"),
                "otp": {
                    "key_bitlength": OneOf(64, 128).set_default(64),
                    "data_type": "numpy.ndarray",
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
        }
    }
}
