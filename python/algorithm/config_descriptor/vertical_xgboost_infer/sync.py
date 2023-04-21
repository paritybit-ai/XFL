from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


vertical_xgboost_infer_sync_rule = {
    "train_info": {
        "train_params": {
            "batch_size_val": All()
        }
    }
}
