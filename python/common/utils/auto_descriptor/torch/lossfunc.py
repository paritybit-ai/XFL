from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


lossfunc = {
    "AdaptiveLogSoftmaxWithLoss": {
        "in_features": All("No default value"),
        "n_classes": All("No default value"),
        "cutoffs": All("No default value"),
        "div_value": Float(4.0),
        "head_bias": Bool(False),
        "device": All(None),
        "dtype": All(None),
        "__rule__": [Required("in_features", "n_classes", "cutoffs"), Optional("head_bias"), Optional("device"), Optional("div_value"), Optional("dtype")]
    },
    "BCELoss": {
        "weight": All(None),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("weight"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "BCEWithLogitsLoss": {
        "weight": All(None),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "pos_weight": All(None),
        "__rule__": [Optional("weight"), Optional("reduce"), Optional("size_average"), Optional("pos_weight"), Optional("reduction")]
    },
    "CTCLoss": {
        "blank": Integer(0),
        "reduction": String("mean"),
        "zero_infinity": Bool(False),
        "__rule__": [Optional("blank"), Optional("zero_infinity"), Optional("reduction")]
    },
    "CosineEmbeddingLoss": {
        "margin": Float(0.0),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("margin"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "CrossEntropyLoss": {
        "weight": All(None),
        "size_average": All(None),
        "ignore_index": Integer(-100),
        "reduce": All(None),
        "reduction": String("mean"),
        "label_smoothing": Float(0.0),
        "__rule__": [Optional("weight"), Optional("reduce"), Optional("size_average"), Optional("ignore_index"), Optional("label_smoothing"), Optional("reduction")]
    },
    "GaussianNLLLoss": {
        "full": Bool(False),
        "eps": Float(1e-06),
        "reduction": String("mean"),
        "__rule__": [Optional("eps"), Optional("full"), Optional("reduction")]
    },
    "HingeEmbeddingLoss": {
        "margin": Float(1.0),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("margin"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "HuberLoss": {
        "reduction": String("mean"),
        "delta": Float(1.0),
        "__rule__": [Optional("delta"), Optional("reduction")]
    },
    "KLDivLoss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "log_target": Bool(False),
        "__rule__": [Optional("log_target"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "L1Loss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "MSELoss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "MarginRankingLoss": {
        "margin": Float(0.0),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("margin"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "MultiLabelMarginLoss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "MultiLabelSoftMarginLoss": {
        "weight": All(None),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("weight"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "MultiMarginLoss": {
        "p": Integer(1),
        "margin": Float(1.0),
        "weight": All(None),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("weight"), Optional("reduce"), Optional("p"), Optional("size_average"), Optional("margin"), Optional("reduction")]
    },
    "NLLLoss": {
        "weight": All(None),
        "size_average": All(None),
        "ignore_index": Integer(-100),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("weight"), Optional("reduce"), Optional("size_average"), Optional("ignore_index"), Optional("reduction")]
    },
    "NLLLoss2d": {
        "weight": All(None),
        "size_average": All(None),
        "ignore_index": Integer(-100),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("weight"), Optional("reduce"), Optional("size_average"), Optional("ignore_index"), Optional("reduction")]
    },
    "PoissonNLLLoss": {
        "log_input": Bool(True),
        "full": Bool(False),
        "size_average": All(None),
        "eps": Float(1e-08),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("reduction"), Optional("reduce"), Optional("size_average"), Optional("full"), Optional("log_input"), Optional("eps")]
    },
    "SmoothL1Loss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "beta": Float(1.0),
        "__rule__": [Optional("beta"), Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "SoftMarginLoss": {
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("size_average"), Optional("reduce"), Optional("reduction")]
    },
    "TripletMarginLoss": {
        "margin": Float(1.0),
        "p": Float(2.0),
        "eps": Float(1e-06),
        "swap": Bool(False),
        "size_average": All(None),
        "reduce": All(None),
        "reduction": String("mean"),
        "__rule__": [Optional("reduction"), Optional("reduce"), Optional("size_average"), Optional("p"), Optional("swap"), Optional("margin"), Optional("eps")]
    },
    "TripletMarginWithDistanceLoss": {
        "distance_function": All(None),
        "margin": Float(1.0),
        "swap": Bool(False),
        "reduction": String("mean"),
        "__rule__": [Optional("reduction"), Optional("distance_function"), Optional("margin"), Optional("swap")]
    }
}
