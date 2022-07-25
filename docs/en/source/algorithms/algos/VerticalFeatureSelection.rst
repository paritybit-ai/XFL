===========================
Vertical Feature Selection
===========================

Introduction
------------
Feature selection is performed according to the iv value and the Pearson correlation coefficient (:ref:`vertical-pearson`) on the vertical federated learning system.
This operator consists of three stages:

1. Vertical Binning Woe IV
2. Vertical Pearson
3. Performing feature selection based on the results of the above two steps.

Parameters List
---------------

The first two stages are vertical_binning_woe_iv and vertical_pearson respectively. For detailed parameter explanation, see Vertical Binning Woe Iv and Vertical Pearson.
The third stage is feature selection, and the specific parameters are defined as follows:

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
        - **path**: ``str`` Folder path of output iv model.
        - **name**: ``str`` File name of output iv model.
        - **type**: ``str`` File type of output iv model.
    - **corr_result**:
        - **path**: ``str`` Folder path of output pearson model.
        - **name**: ``str`` File name of output pearson model.
        - **type**: ``str`` File type of output pearson model.

**output**:
    - **model**:
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
        - **type**: ``str`` File type of output model.
    - **trainset**:
        - **type**: ``str`` Train dataset type after feature selection, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset after feature selection.
        - **name**: ``str`` If type is `csv`, file name of train dataset after feature selection.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**:
        - **type**: ``str`` Validation dataset type after feature selection, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of validation dataset after feature selection.
        - **name**: ``str`` If type is `csv`, file name of validation dataset after feature selection.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **filter_params**:
            - **common**:
                - **metrics**: ``str`` Metrics evaluation parameters.
                - **filter_method**: ``str`` Feature filtering method.
                - **threshold**: ``float`` Feature filtering threshold.
            - **correlation**:
                - **sort_metric**: ``str`` Metric to sort on.
                - **correlation_threshold**: ``float`` Correlation filtering threshold.