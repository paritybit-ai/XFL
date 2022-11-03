====================
Local Normalization
====================

Introduction
------------

Local normalization is a module that normalizes the features of the local data (so that the p-norm ( `p-norm` ) of each feature after normalization takes a value of 1).

Specifically, for the feature matrix :math:`X`, perform `Local Normalization` transformation to obtain a new feature matrix :math:`\tilde{X}`. 
If features are normalized（`axis=0`）, then :math:`||\tilde{X}_{.j}||_p = 1\text{, }\forall j \in {1,\dots, m}`. 
If samples are normalized, then :math:`||\tilde{X}_{i.}||_p = 1\text{, }\forall i \in {1,\dots, n}`.


Parameter List
--------------

**identity**: ``str`` The role of each participant in federated learning, should be `label_trainer` or `trainer`.

**model_info**:
    - **name**: ``str``  Model name, should be `local_normalization`.

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
        - **norm**: ``str`` The norm to use (``"l1"``/``"l2"``/``"max"``).
        - **axis**: ``int`` Axis along which the normalization is applied. 1 for row normalization, 0 for column normalization.
        - **feature_norm**: ``map`` Fine-grained configuration for column normalization(axis = 0). The format is defined as: ``{column_1: {"norm": norm_1}, ..., column_N: {"norm": norm_N}}``.