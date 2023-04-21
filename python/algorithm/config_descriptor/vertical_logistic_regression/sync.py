from x_types import String, Bool, Integer, Float, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_logistic_regression_sync_rule = {
    "train_info": {
        "interaction_params": All(),
        "train_params": {
            "global_epoch": All(),
            "batch_size": All(),
            "encryption": All(),
            "optimizer": All(),
            "random_seed": All()
        }
    }
}


