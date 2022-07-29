=============================
Vertical Logistic Regression
=============================

Introduction
------------

The reaization of vertical logistic regression algorithm is based on [Yang2019]_ .

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_logistic_regression`.
    - **config**:
        - **input_dim**: ``int`` Size of each input sample. 
        - **bias**: ``bool`` Whether having a learnable additive bias.


**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**:
        - **type**: ``str`` Validation dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of validation dataset.
        - **name**: ``bool`` If type is `csv`, file name of validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.

**output**:  
    - **model**: 
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **metrics**:  
        - **path**: ``str`` If type is `csv`, folder path of metrics.
        - **header**: ``bool`` Whether having a header.
    - **evaluations**:  
        - **path**: ``str`` If type is `csv`, folder path of evaluations.
        - **header**: ``bool`` Whether having a header.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **interaction_params**:  
        - **save_frequency**: ``int`` Frequency to save model, set to -1 for not saving model.
        - **save_probabilities**: ``bool`` Whether to save probabilities bins.
        - **save_probabilities_bins_number**: ``int`` Number of bins.
        - **write_training_prediction**: ``bool`` Whether to save the predictions on train dataset.
        - **echo_training_metrics**: ``bool`` Whether to output metrics on train dataset.
        - **write_validation_prediction**: ``bool`` Whether to save predictions on validation dataset.

    - **params**:  
        - **global_epoch**: ``int`` Global training epoch.
        - **batch_size**: ``int`` Batch size of samples in global process.
        - **aggregation_config**:
            - **encryption**:
                - **method**: ``str`` Encryption method, recommend "ckks".
                - **poly_modulus_degree**: ``int``  Polynomial modulus degree.
                - **coeff_mod_bit_sizes**: ``list``  Coefficient modulus sizes.
                - **global_scale_bit_size**: ``int`` Global scale factor bit size.

        - **optimizer_config**: 
            - **lr**: ``float`` Learning rate.
            - **p**: ``int`` Regularization parameter, "0"/"1"/"2" stands for no regularization/l1 regularization/l2 regularization respectively.
            - **alpha**: ``float`` Penalty coefficient.

        - **lossfunc_config**:
            - **method**: Loss function configuration, support `BCEWithLogitsLoss`.
        - **metric_config**:
            - **accuracy**: Accuracy.
            - **precision**: Precision.
            - **recall**: Recall.
            - **f1_score**: F1 score.
            - **auc**: Area Under Curve.
            - **ks**: Kolmogorov-Smirnov (KS) Statistics.

        - **early_stopping**:
            - **key**: ``str`` Variable to be monitored.
            - **patience**: ``int`` Number of epochs with no improvement after which training will be stopped.
            - **delta**: ``int`` Minimum change of the key to qualify as an improvement.

        - **extra_config**:
            - **shuffle_seed**: ``int`` Random seed.


.. [Yang2019] Yang S, Ren B, Zhou X, et al. Parallel distributed logistic regression for vertical federated learning without third-party coordinator[J]. arXiv preprint arXiv:1911.09824, 2019.