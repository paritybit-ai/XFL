=======================
Local Standard Scaler
=======================

Introduction
------------

Local standard scaler is a module that standardize each column of features by subtracting the mean then and then scaling to unit variance. 

Parameter List
--------------

**identity**: ``str`` The role of each participant in federated learning, should be `label_trainer` or `trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_standard_scaler`.

**input**:
    - **trainset**: 
        - **type**: ``str`` Train dataset type, currently supported is `csv`.
        - **path**: ``str`` The folder path of train dataset.
        - **name**: ``str`` The file name of train dataset.
        - **has_id**: ``bool`` Whether the dataset has id column.
        - **has_label**: ``bool`` Whether the dataset has label column.
    - **valset**:
        - **type**: ``str`` Validation dataset type, currently supported is `csv`.
        - **path**: ``str`` The folder path of validation dataset.
        - **name**: ``str`` The file name of validation dataset.
        - **has_id**: ``bool`` Whether the dataset has id column.
        - **has_label**: ``bool`` Whether the dataset has label column.
**output**:
    - **path**: ``str`` Output folder path.
    - **model**:
        - **name**: ``str`` File name of output model.
    - **trainset**:
        - **name**: ``str`` File name of output train dataset.
    - **valset**:
        - **name**: ``str`` File name of output validation dataset.

**train_info**:  
    - **train_params**:
        - **with_mean**: ``bool`` If True, center the data before scaling.
        - **with_std**: ``bool`` If True, scale the data to unit standard deviation.
        - **feature_standard**: ``map`` Fine-grained configuration for standardization. The format is defined as: ``{column_1: {"with_mean": True/False, "with_std": True/False}, ..., column_N: {"with_mean": True/False, "with_std": True/False}}``.
