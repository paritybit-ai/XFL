=======================
Local Standard Scaler
=======================

Introduction
------------

Local standard scaler is a module that standardize each column of features by subtracting the mean then and then scaling to unit variance. 

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
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **trainset**: 
        - **path**: ``str`` Folder path of output train dataset.
        - **name**: ``str`` File name of output train dataset.
    - **valset**: 
        - **path**: ``str`` Folder path of output validation dataset.
        - **name**: ``str`` File name of output validation dataset.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:  Parameters `with_mean` and `with_std` is valid for all data except for the data involved in `feature_standardize_config`.
        - **with_mean**: ``bool`` If True, center the data before scaling by column.
        - **with_std**: ``bool`` If True, scale the data to unit standard deviation by column.
        - **feature_standardize_config**: ``repeated<map>`` Fine-grained configuration for standardization. The format is defined as: {column name: {with_mean: `bool`, with_std: `bool`},..., column name: {with_mean: `bool`, with_std: `bool`}}.