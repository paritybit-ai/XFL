from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


lr_scheduler = {
    "ConstantLR": {
        "factor": Float(0.3333333333333333),
        "total_iters": Integer(5),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Optional("last_epoch"), Optional("total_iters"), Optional("factor"), Optional("verbose")]
    },
    "CosineAnnealingLR": {
        "T_max": All("No default value"),
        "eta_min": Integer(0),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("T_max"), Optional("eta_min"), Optional("last_epoch"), Optional("verbose")]
    },
    "CosineAnnealingWarmRestarts": {
        "T_0": All("No default value"),
        "T_mult": Integer(1),
        "eta_min": Integer(0),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("T_0"), Optional("eta_min"), Optional("last_epoch"), Optional("verbose"), Optional("T_mult")]
    },
    "CyclicLR": {
        "base_lr": All("No default value"),
        "max_lr": All("No default value"),
        "step_size_up": Integer(2000),
        "step_size_down": All(None),
        "mode": String("triangular"),
        "gamma": Float(1.0),
        "scale_fn": All(None),
        "scale_mode": String("cycle"),
        "cycle_momentum": Bool(True),
        "base_momentum": Float(0.8),
        "max_momentum": Float(0.9),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("base_lr", "max_lr"), Optional("mode"), Optional("base_momentum"), Optional("last_epoch"), Optional("gamma"), Optional("verbose"), Optional("scale_fn"), Optional("max_momentum"), Optional("step_size_down"), Optional("step_size_up"), Optional("cycle_momentum"), Optional("scale_mode")]
    },
    "ExponentialLR": {
        "gamma": All("No default value"),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("gamma"), Optional("last_epoch"), Optional("verbose")]
    },
    "LambdaLR": {
        "lr_lambda": All("No default value"),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("lr_lambda"), Optional("last_epoch"), Optional("verbose")]
    },
    "LinearLR": {
        "start_factor": Float(0.3333333333333333),
        "end_factor": Float(1.0),
        "total_iters": Integer(5),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Optional("total_iters"), Optional("end_factor"), Optional("last_epoch"), Optional("start_factor"), Optional("verbose")]
    },
    "MultiStepLR": {
        "milestones": All("No default value"),
        "gamma": Float(0.1),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("milestones"), Optional("last_epoch"), Optional("gamma"), Optional("verbose")]
    },
    "MultiplicativeLR": {
        "lr_lambda": All("No default value"),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("lr_lambda"), Optional("last_epoch"), Optional("verbose")]
    },
    "OneCycleLR": {
        "max_lr": All("No default value"),
        "total_steps": All(None),
        "epochs": All(None),
        "steps_per_epoch": All(None),
        "pct_start": Float(0.3),
        "anneal_strategy": String("cos"),
        "cycle_momentum": Bool(True),
        "base_momentum": Float(0.85),
        "max_momentum": Float(0.95),
        "div_factor": Float(25.0),
        "final_div_factor": Float(10000.0),
        "three_phase": Bool(False),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("max_lr"), Optional("div_factor"), Optional("final_div_factor"), Optional("base_momentum"), Optional("last_epoch"), Optional("verbose"), Optional("pct_start"), Optional("cycle_momentum"), Optional("epochs"), Optional("max_momentum"), Optional("steps_per_epoch"), Optional("total_steps"), Optional("three_phase"), Optional("anneal_strategy")]
    },
    "ReduceLROnPlateau": {
        "mode": String("min"),
        "factor": Float(0.1),
        "patience": Integer(10),
        "threshold": Float(0.0001),
        "threshold_mode": String("rel"),
        "cooldown": Integer(0),
        "min_lr": Integer(0),
        "eps": Float(1e-08),
        "verbose": Bool(False),
        "__rule__": [Optional("mode"), Optional("threshold_mode"), Optional("threshold"), Optional("patience"), Optional("verbose"), Optional("eps"), Optional("cooldown"), Optional("min_lr"), Optional("factor")]
    },
    "SequentialLR": {
        "schedulers": All("No default value"),
        "milestones": All("No default value"),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("schedulers", "milestones"), Optional("last_epoch"), Optional("verbose")]
    },
    "StepLR": {
        "step_size": All("No default value"),
        "gamma": Float(0.1),
        "last_epoch": Integer(-1),
        "verbose": Bool(False),
        "__rule__": [Required("step_size"), Optional("last_epoch"), Optional("gamma"), Optional("verbose")]
    }
}
