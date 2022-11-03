=======================
Local Data Statistic
=======================

Introduction
------------

Local data statistic is a module that calculates several statistical indicators of features, including mean, maximum and minimum, median, etc.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer` or `trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_data_statistic`.

**input**:
    - **dataset**:
        - **type**: ``str`` Input dataset type, support `csv`.
        - **path**: ``str`` Folder path of input dataset.
        - **name**: ``str`` File name of input dataset.
        - **has_label**: ``bool`` Whether dataset has label column.
        - **has_id**: ``bool`` Whether dataset has id.

**output**:
    - **path**: ``str`` Folder path of output.
    - **summary**:
        - **name**: ``str`` File name of output result.

**train_info**:  
    - **train_params**:
        - **quantile**: ``list`` The quantile(s) to compute, default [0.25, 0.5, 0.75].
