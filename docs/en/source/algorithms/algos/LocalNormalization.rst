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

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str``  Model name, should be `local_normalization`.
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
    - **params**: Parameters `norm` and `axis` is valid for all data except for the data involved in `featureNormalizeConfig`.
        - **norm**: ``str`` Types of normalization (``"l1"``/``"l2"``/``"max"``).
        - **axis**: ``int`` Axis along which normalization is applied. 1 for row normalization, 0 for column normalization.
        - **featureNormalizeConfig**: ``map`` Fine-grained configuration for column normalization(axis = 0). The format is defined as: {column name: {"norm": type of normalization}, ..., column name: {"norm": type of normalization}}.