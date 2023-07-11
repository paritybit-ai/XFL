from common.checker.x_types import String, Bool, Integer, Float, Any
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from common.utils.auto_descriptor.torch.optimizer import optimizer
from common.utils.auto_descriptor.torch.lr_scheduler import lr_scheduler
from common.utils.auto_descriptor.torch.lossfunc import lossfunc
from common.utils.auto_descriptor.torch.metrics import metrics
from common.utils.utils import update_dict
from algorithm.core.metrics import metric_dict


horizontal_chatglm_assist_trainer_rule = {
    "identity": "assist_trainer",
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
            "aggregation": {
                "agg_steps": float(0.2)
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
            },
            "peft": {
                "__rule__": OneOf("LORA", "PREFIX_TUNING", "ADALOARA"),
                "LORA": {
                    "task_type": "CAUSAL_LM",
                    "r": Integer(8),
                    "target_modules": ["query_key_value"],
                    "lora_alpha": Integer(32),
                    "lora_dropout": Float(0.1),
                    "fan_in_fan_out": Bool(False),
                    "bias": OneOf("none", "all", "loral_only"),
                    "modules_to_save": None
                },
                "PREFIX_TUNING": {
                    "task_type": "CAUSAL_LM",
                    "pre_seq_len": Integer(20),
                    "prefix_projection": Bool(False)
                },
                "ADALORA": {
                    "task_type": "CAUSAL_LM",
                    "r": Integer(8),
                    "target_modules": ["query_key_value"],
                    "lora_alpha": Integer(32),
                    "lora_dropout": Float(0.1),
                    "fan_in_fan_out": Bool(False),
                    "bias": OneOf("none", "all", "loral_only"),
                    "modules_to_save": None,
                    "target_r": Integer(8),
                    "init_r": Integer(12),
                    "tinit": Integer(0),
                    "tfinal": Integer(0),
                    "deltaT": Integer(1),
                    "beta1": Float(0.85),
                    "beta2": Float(0.85),
                    "orth_reg_weight": Float(0.5)
                }
            },
            "trainer": {
                "per_device_train_batch_size": Integer(1),
                "gradient_accumulation_steps": Integer(4),
                "learning_rate": Float(1e-4),
                "weight_decay": Float(0),
                "adam_beta1": Float(0.9),
                "adam_beta2": Float(0.999),
                "adam_epsilon": Float(1e-8),
                "max_grad_norm": Float(1.0),
                "num_train_epochs": Integer(2),
                "save_strategy": OneOf("steps", "no"),
                "torch_compile": Bool(False),
                "no_cuda": Bool(False),
                "seed": Integer(42)
            },
            "dataset": {
                "max_src_length": Integer(100),
                "max_dst_length": Integer(100),
                "ignore_pad_token_for_loss": Bool(True)
            }
        }
    }
}
