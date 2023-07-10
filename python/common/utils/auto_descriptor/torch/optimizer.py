from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


optimizer = {
    "ASGD": {
        "lr": Float(0.01),
        "lambd": Float(0.0001),
        "alpha": Float(0.75),
        "t0": Float(1000000.0),
        "weight_decay": Integer(0),
        "__rule__": [Optional("lr"), Optional("alpha"), Optional("t0"), Optional("lambd"), Optional("weight_decay")]
    },
    "Adadelta": {
        "lr": Float(1.0),
        "rho": Float(0.9),
        "eps": Float(1e-06),
        "weight_decay": Integer(0),
        "__rule__": [Optional("lr"), Optional("weight_decay"), Optional("rho"), Optional("eps")]
    },
    "Adagrad": {
        "lr": Float(0.01),
        "lr_decay": Integer(0),
        "weight_decay": Integer(0),
        "initial_accumulator_value": Integer(0),
        "eps": Float(1e-10),
        "__rule__": [Optional("lr"), Optional("weight_decay"), Optional("initial_accumulator_value"), Optional("lr_decay"), Optional("eps")]
    },
    "Adam": {
        "lr": Float(0.001),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "weight_decay": Integer(0),
        "amsgrad": Bool(False),
        "maximize": Bool(False),
        "__rule__": [Optional("lr"), Optional("eps"), Optional("maximize"), Optional("betas"), Optional("amsgrad"), Optional("weight_decay")]
    },
    "AdamW": {
        "lr": Float(0.001),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "weight_decay": Float(0.01),
        "amsgrad": Bool(False),
        "maximize": Bool(False),
        "__rule__": [Optional("lr"), Optional("eps"), Optional("maximize"), Optional("betas"), Optional("amsgrad"), Optional("weight_decay")]
    },
    "Adamax": {
        "lr": Float(0.002),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "weight_decay": Integer(0),
        "__rule__": [Optional("lr"), Optional("weight_decay"), Optional("eps"), Optional("betas")]
    },
    "LBFGS": {
        "lr": Integer(1),
        "max_iter": Integer(20),
        "max_eval": All(None),
        "tolerance_grad": Float(1e-07),
        "tolerance_change": Float(1e-09),
        "history_size": Integer(100),
        "line_search_fn": All(None),
        "__rule__": [Optional("tolerance_change"), Optional("lr"), Optional("history_size"), Optional("tolerance_grad"), Optional("max_eval"), Optional("line_search_fn"), Optional("max_iter")]
    },
    "NAdam": {
        "lr": Float(0.002),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "weight_decay": Integer(0),
        "momentum_decay": Float(0.004),
        "__rule__": [Optional("lr"), Optional("eps"), Optional("betas"), Optional("momentum_decay"), Optional("weight_decay")]
    },
    "RAdam": {
        "lr": Float(0.001),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "weight_decay": Integer(0),
        "__rule__": [Optional("lr"), Optional("weight_decay"), Optional("eps"), Optional("betas")]
    },
    "RMSprop": {
        "lr": Float(0.01),
        "alpha": Float(0.99),
        "eps": Float(1e-08),
        "weight_decay": Integer(0),
        "momentum": Integer(0),
        "centered": Bool(False),
        "__rule__": [Optional("lr"), Optional("alpha"), Optional("eps"), Optional("momentum"), Optional("centered"), Optional("weight_decay")]
    },
    "Rprop": {
        "lr": Float(0.01),
        "etas": [
                Float(0.5),
                Float(1.2),
            ],
        "step_sizes": [
                Float(1e-06),
                Integer(50),
            ],
        "__rule__": [Optional("lr"), Optional("etas"), Optional("step_sizes")]
    },
    "SGD": {
        "lr": All("No default value"),
        "momentum": Integer(0),
        "dampening": Integer(0),
        "weight_decay": Integer(0),
        "nesterov": Bool(False),
        "maximize": Bool(False),
        "__rule__": [Required("lr"), Optional("dampening"), Optional("nesterov"), Optional("maximize"), Optional("weight_decay"), Optional("momentum")]
    },
    "SparseAdam": {
        "lr": Float(0.001),
        "betas": [
                Float(0.9),
                Float(0.999),
            ],
        "eps": Float(1e-08),
        "__rule__": [Optional("lr"), Optional("betas"), Optional("eps")]
    }
}
