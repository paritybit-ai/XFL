=======================
Local Standard Scaler
=======================

Introduction
------------

Local standard scaler is a module that standardize each column of features by subtracting the mean and dividing by 
the standard deviation to have a mean of 0 and standard deviation of 1. Parameter `train_info` defines different configuration information for each participant 
during the training process.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_standard_scaler`.
    - **config**: ``map`` Model configuration, no need to config here.

**input**:
    - **trainset**: 
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``str`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**: 
        - **type**: ``str`` Validation dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of validation dataset.
        - **name**: ``str`` If type is `csv`, file name of validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
**output**:
    - **model**:
        - **type**: ``str`` Model output format, support "file".
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **trainset**: 
        - **type**: ``str`` Output train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of output train dataset.
        - **name**: ``str`` If type is `csv`, file name of output train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**: 
        - **type**: ``str`` Output validation dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of output validation dataset.
        - **name**: ``str`` If type is `csv`, file name of output validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:  
        - **with_mean**: ``str`` Whether the mean is provided, if not set to 0.
        - **with_std**: ``int`` Whether the std is provided, if not set to 1.
        - **feature_standardize_config**: ``repeated<map>`` Different standardization settings for each column, of which the format is: {column name: {with_mean: `bool`, with_std: `bool`},...,column name: {with_mean: `bool`, with_std: `bool`}}.