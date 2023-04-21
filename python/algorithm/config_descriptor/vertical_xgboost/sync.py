from x_types import String, Bool, Integer, Float, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_xgboost_sync_rule = {
    "train_info": {
        "interaction_params": All(),
        "train_params": {
            "lossfunc": All(),
            "num_trees": All(),
            "num_bins": All(),
            "batch_size_val": All(),
            "downsampling": {
                "row": {
                    "run_goss": All()
                }
            },
            "encryption": All()
        }
    }
}


