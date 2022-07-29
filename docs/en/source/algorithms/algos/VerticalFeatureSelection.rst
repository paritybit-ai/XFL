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

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_feature_selection`.

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
    - **iv_result**:
        - **path**: ``str`` Folder path of the iv_result from `VerticalBinningWoeIV`.
        - **name**: ``str`` File name.
    - **corr_result**:
        - **path**: ``str`` Folder path of the result from `VerticalPearson`.
        - **name**: ``str`` File namel.

**output**:
    - **model**:
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
        - **type**: ``str`` File type, support "csv".
    - **trainset**:
        - **path**: ``str`` Folder path of train dataset after feature selection.
        - **name**: ``str`` File name of train dataset after feature selection.
    - **valset**:
        - **path**: ``str`` Folder path of validation dataset after feature selection.
        - **name**: ``str`` File name of validation dataset after feature selection.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **filter_params**:
            - **common**:
                - **metrics**: ``str`` Metric type, support `iv`.
                - **filter_method**: ``str`` Feature filtering method, support `threshold`.
                - **threshold**: ``float`` Feature filtering threshold if filter_method is `threshold`.
            - **correlation**:
                - **sort_metric**: ``str`` Metric type for sorting, support `iv`.
                - **correlation_threshold**: ``float`` Correlation threshold.