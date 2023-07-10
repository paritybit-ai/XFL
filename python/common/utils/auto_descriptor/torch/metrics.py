from common.checker.x_types import String, Bool, Integer, Float, Any, All
from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional


metrics = {
    "acc": {
        "normalize": Bool(True),
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight"), Optional("normalize")]
    },
    "adjusted_mutual_info_score": {
        "average_method": String("arithmetic"),
        "__rule__": [Optional("average_method")]
    },
    "adjusted_rand_score": {
    },
    "auc": {
    },
    "average_precision_score": {
        "average": String("macro"),
        "pos_label": Integer(1),
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight"), Optional("pos_label"), Optional("average")]
    },
    "balanced_accuracy_score": {
        "sample_weight": All(None),
        "adjusted": Bool(False),
        "__rule__": [Optional("sample_weight"), Optional("adjusted")]
    },
    "brier_score_loss": {
        "sample_weight": All(None),
        "pos_label": All(None),
        "__rule__": [Optional("sample_weight"), Optional("pos_label")]
    },
    "calinski_harabasz_score": {
    },
    "classification_report": {
        "labels": All(None),
        "target_names": All(None),
        "sample_weight": All(None),
        "digits": Integer(2),
        "output_dict": Bool(False),
        "zero_division": String("warn"),
        "__rule__": [Optional("target_names"), Optional("digits"), Optional("labels"), Optional("sample_weight"), Optional("output_dict"), Optional("zero_division")]
    },
    "completeness_score": {
    },
    "confusion_matrix": {
        "labels": All(None),
        "sample_weight": All(None),
        "normalize": All(None),
        "__rule__": [Optional("sample_weight"), Optional("normalize"), Optional("labels")]
    },
    "consensus_score": {
        "similarity": String("jaccard"),
        "__rule__": [Optional("similarity")]
    },
    "coverage_error": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "d2_tweedie_score": {
        "sample_weight": All(None),
        "power": Integer(0),
        "__rule__": [Optional("sample_weight"), Optional("power")]
    },
    "davies_bouldin_score": {
    },
    "dcg_score": {
        "k": All(None),
        "log_base": Integer(2),
        "sample_weight": All(None),
        "ignore_ties": Bool(False),
        "__rule__": [Optional("sample_weight"), Optional("ignore_ties"), Optional("k"), Optional("log_base")]
    },
    "det_curve": {
        "pos_label": All(None),
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight"), Optional("pos_label")]
    },
    "euclidean_distances": {
        "Y_norm_squared": All(None),
        "squared": Bool(False),
        "X_norm_squared": All(None),
        "__rule__": [Optional("squared"), Optional("X_norm_squared"), Optional("Y_norm_squared")]
    },
    "explained_variance_score": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "__rule__": [Optional("sample_weight"), Optional("multioutput")]
    },
    "f1_score": {
        "labels": All(None),
        "pos_label": Integer(1),
        "average": String("binary"),
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Optional("labels"), Optional("sample_weight"), Optional("zero_division"), Optional("pos_label"), Optional("average")]
    },
    "fbeta_score": {
        "beta": All("No default value"),
        "labels": All(None),
        "pos_label": Integer(1),
        "average": String("binary"),
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Required("beta"), Optional("sample_weight"), Optional("pos_label"), Optional("labels"), Optional("zero_division"), Optional("average")]
    },
    "fowlkes_mallows_score": {
        "sparse": Bool(False),
        "__rule__": [Optional("sparse")]
    },
    "hamming_loss": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "homogeneity_completeness_v_measure": {
        "beta": Float(1.0),
        "__rule__": [Optional("beta")]
    },
    "homogeneity_score": {
    },
    "jaccard_score": {
        "labels": All(None),
        "pos_label": Integer(1),
        "average": String("binary"),
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Optional("labels"), Optional("sample_weight"), Optional("zero_division"), Optional("pos_label"), Optional("average")]
    },
    "label_ranking_average_precision_score": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "label_ranking_loss": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "log_loss": {
        "eps": Float(1e-15),
        "normalize": Bool(True),
        "sample_weight": All(None),
        "labels": All(None),
        "__rule__": [Optional("sample_weight"), Optional("normalize"), Optional("labels"), Optional("eps")]
    },
    "matthews_corrcoef": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "max_error": {
    },
    "mae": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "__rule__": [Optional("sample_weight"), Optional("multioutput")]
    },
    "mape": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "__rule__": [Optional("sample_weight"), Optional("multioutput")]
    },
    "mean_gamma_deviance": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "mean_pinball_loss": {
        "sample_weight": All(None),
        "alpha": Float(0.5),
        "multioutput": String("uniform_average"),
        "__rule__": [Optional("sample_weight"), Optional("alpha"), Optional("multioutput")]
    },
    "mean_poisson_deviance": {
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight")]
    },
    "mse": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "squared": Bool(True),
        "__rule__": [Optional("sample_weight"), Optional("multioutput"), Optional("squared")]
    },
    "mean_squared_log_error": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "squared": Bool(True),
        "__rule__": [Optional("sample_weight"), Optional("multioutput"), Optional("squared")]
    },
    "mean_tweedie_deviance": {
        "sample_weight": All(None),
        "power": Integer(0),
        "__rule__": [Optional("sample_weight"), Optional("power")]
    },
    "median_ae": {
        "multioutput": String("uniform_average"),
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight"), Optional("multioutput")]
    },
    "multilabel_confusion_matrix": {
        "sample_weight": All(None),
        "labels": All(None),
        "samplewise": Bool(False),
        "__rule__": [Optional("sample_weight"), Optional("samplewise"), Optional("labels")]
    },
    "mutual_info_score": {
        "contingency": All(None),
        "__rule__": [Optional("contingency")]
    },
    "nan_euclidean_distances": {
        "squared": Bool(False),
        "missing_values": Float(None),
        "copy": Bool(True),
        "__rule__": [Optional("squared"), Optional("missing_values"), Optional("copy")]
    },
    "ndcg_score": {
        "k": All(None),
        "sample_weight": All(None),
        "ignore_ties": Bool(False),
        "__rule__": [Optional("sample_weight"), Optional("ignore_ties"), Optional("k")]
    },
    "normalized_mutual_info_score": {
        "average_method": String("arithmetic"),
        "__rule__": [Optional("average_method")]
    },
    "pair_confusion_matrix": {
    },
    "pairwise_distances": {
        "metric": String("euclidean"),
        "n_jobs": All(None),
        "force_all_finite": Bool(True),
        "__rule__": [Optional("force_all_finite"), Optional("n_jobs"), Optional("metric")]
    },
    "pairwise_distances_argmin": {
        "axis": Integer(1),
        "metric": String("euclidean"),
        "metric_kwargs": All(None),
        "__rule__": [Optional("axis"), Optional("metric_kwargs"), Optional("metric")]
    },
    "pairwise_distances_argmin_min": {
        "axis": Integer(1),
        "metric": String("euclidean"),
        "metric_kwargs": All(None),
        "__rule__": [Optional("axis"), Optional("metric_kwargs"), Optional("metric")]
    },
    "pairwise_distances_chunked": {
        "reduce_func": All(None),
        "metric": String("euclidean"),
        "n_jobs": All(None),
        "working_memory": All(None),
        "__rule__": [Optional("working_memory"), Optional("reduce_func"), Optional("n_jobs"), Optional("metric")]
    },
    "pairwise_kernels": {
        "metric": String("linear"),
        "filter_params": Bool(False),
        "n_jobs": All(None),
        "__rule__": [Optional("filter_params"), Optional("n_jobs"), Optional("metric")]
    },
    "precision_recall_fscore_support": {
        "beta": Float(1.0),
        "labels": All(None),
        "pos_label": Integer(1),
        "average": All(None),
        "warn_for": [
                String("precision"),
                String("recall"),
                String("f-score"),
            ],
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Optional("warn_for"), Optional("beta"), Optional("labels"), Optional("sample_weight"), Optional("zero_division"), Optional("pos_label"), Optional("average")]
    },
    "precision": {
        "labels": All(None),
        "pos_label": Integer(1),
        "average": String("binary"),
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Optional("labels"), Optional("sample_weight"), Optional("zero_division"), Optional("pos_label"), Optional("average")]
    },
    "r2": {
        "sample_weight": All(None),
        "multioutput": String("uniform_average"),
        "__rule__": [Optional("sample_weight"), Optional("multioutput")]
    },
    "rand_score": {
    },
    "recall": {
        "labels": All(None),
        "pos_label": Integer(1),
        "average": String("binary"),
        "sample_weight": All(None),
        "zero_division": String("warn"),
        "__rule__": [Optional("labels"), Optional("sample_weight"), Optional("zero_division"), Optional("pos_label"), Optional("average")]
    },
    "auc": {
        "average": String("macro"),
        "sample_weight": All(None),
        "max_fpr": All(None),
        "multi_class": String("raise"),
        "labels": All(None),
        "__rule__": [Optional("max_fpr"), Optional("labels"), Optional("sample_weight"), Optional("multi_class"), Optional("average")]
    },
    "roc_curve": {
        "pos_label": All(None),
        "sample_weight": All(None),
        "drop_intermediate": Bool(True),
        "__rule__": [Optional("sample_weight"), Optional("drop_intermediate"), Optional("pos_label")]
    },
    "silhouette_samples": {
        "metric": String("euclidean"),
        "__rule__": [Optional("metric")]
    },
    "silhouette_score": {
        "metric": String("euclidean"),
        "sample_size": All(None),
        "random_state": All(None),
        "__rule__": [Optional("sample_size"), Optional("random_state"), Optional("metric")]
    },
    "top_k_accuracy_score": {
        "k": Integer(2),
        "normalize": Bool(True),
        "sample_weight": All(None),
        "labels": All(None),
        "__rule__": [Optional("sample_weight"), Optional("normalize"), Optional("k"), Optional("labels")]
    },
    "v_measure_score": {
        "beta": Float(1.0),
        "__rule__": [Optional("beta")]
    },
    "zero_one_loss": {
        "normalize": Bool(True),
        "sample_weight": All(None),
        "__rule__": [Optional("sample_weight"), Optional("normalize")]
    },
    "ks": {
    },
    "rmse": {
    }
}
