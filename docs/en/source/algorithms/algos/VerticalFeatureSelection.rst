===========================
Vertical Feature Selection
===========================

Introduction
------------

Feature selection relies on the output of algorithm `VerticalBinningWoeIV` and `VerticalPearson`.

The operator consists of three stages:

1. Vertical Binning Woe IV
2. Vertical Pearson
3. Performing feature selection based on the results of the above two steps.


Parameters List
---------------

The parameters given below is only for feature selection.

**identity**: ``str`` The role of each participant in federated learning, should be `label_trainer` or `trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_feature_selection`.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, currently supported is `csv`.
        - **path**: ``str`` The folder path of train dataset.
        - **name**: ``str`` The file name of train dataset.
        - **has_id**: ``bool`` Whether the dataset has id column.
        - **has_label**: ``bool`` Whether the dataset has label column.
    - **valset**:
        - **type**: ``str`` Train dataset type, currently supported is `csv`.
        - **path**: ``str`` The folder path of train dataset.
        - **name**: ``str`` The file name of train dataset.
        - **has_id**: ``bool`` Whether the dataset has id column.
        - **has_label**: ``bool`` Whether the dataset has label column.
    - **iv_result**:
        - **path**: ``str`` Folder path of the iv_result from `VerticalBinningWoeIV`.
        - **name**: ``str`` Model file name.
    - **corr_result**:
        - **path**: ``str`` Folder path of the result from `VerticalPearson`.
        - **name**: ``str`` Model file name.

**output**:
    - **path**: ``str`` Output folder path.
    - **model**:
        - **name**: ``str`` File name of output model.
    - **trainset**:
        - **name**: ``str`` File name of train dataset after feature selection.
    - **valset**:
        - **name**: ``str`` File name of validation dataset after feature selection.

**train_info**:
    - **train_params**:
        - **filter**:
            - **common**:
                - **metrics**: ``str`` Metric type, currently supported is `iv`.
                - **filter_method**: ``str`` Feature filtering method, currently supported is `threshold`.
                - **threshold**: ``float`` Threshold for filtering if the filter_method is `threshold`.
            - **correlation**:
                - **sort_metric**: ``str`` Sorting (descending) metric before the correlated filtering, currently supported is `iv`.
                - **correlation_threshold**: ``float`` Threshold for correlated filtering.