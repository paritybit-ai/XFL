=============================
Vertical Poisson Regression
=============================

Introduction
------------

The realization of vertical poisson regression algorithm.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_poisson_regression`.

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
    - **pretrained_model**: ``map``
        - **path**: ``str`` Pretrained model path. 
        - **name**: ``str`` Pretrained model name.

**output**:  
    - **path**: ``str`` Folder path of output.
    - **model**:
        - **name**: ``str`` File name of output model.
    - **metric_train**:
        - **name**: ``str`` File name of trainset metrics.
    - **metric_val**:
        - **name**: ``str`` File name of valset metrics.
    - **prediction_train**:
        - **name**: ``str`` File name of trainset prediction.
    - **prediction_val**:
        - **name**: ``str`` File name of valset prediction.
    - **feature_importance**:
        - **name**: ``str`` File name of feature importance table.

**train_info**:  
    - **interaction_params**:  
        - **save_frequency**: ``int`` Frequency to save model, set to -1 for not saving model.
        - **write_training_prediction**: ``bool`` Whether to save the predictions on train dataset.
        - **echo_training_metrics**: ``bool`` Whether to output metrics on train dataset.
        - **write_validation_prediction**: ``bool`` Whether to save predictions on validation dataset.

    - **train_params**:  
        - **global_epoch**: ``int`` Global training epoch.
        - **batch_size**: ``int`` Batch size of samples in global process.
        - **encryption**: ``map`` Can choose either "ckks" or "paillier".
            - **ckks**: ``map``
                - **poly_modulus_degree**: ``int``  Polynomial modulus degree.
                - **coeff_mod_bit_sizes**: ``list``  Coefficient modulus sizes.
                - **global_scale_bit_size**: ``int`` Global scale factor bit size.
            - **paillier**:
                - **key_bit_size**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                - **precision**: ``int`` Precision.
                - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multicore for computing.

        - **optimizer**: 
            - **lr**: ``float`` Learning rate.
            - **p**: ``int`` Regularization parameter, "0"/"1"/"2" stands for no regularization/l1 regularization/l2 regularization respectively.
            - **alpha**: ``float`` Penalty coefficient.

        - **metric**: ``map`` Metrics to output, all the keys are optional.
            - **mse**: ``map`` The mean squared error , support {}.
            - **mape**: ``map`` The mean absolute percentage error, support {}.
            - **mae**: ``map`` Mean absolute error, support {}.
            - **rmse**: ``map`` Root mean square error, support {}.

        - **early_stopping**:
            - **key**: ``str`` Variable to be monitored.
            - **patience**: ``int`` Number of epochs with no improvement after which training will be stopped.
            - **delta**: ``int`` Minimum change of the key to qualify as an improvement.

        - **random_seed**: ``int`` Random seed, accept None.