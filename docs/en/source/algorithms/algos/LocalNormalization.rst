====================
Local Normalization
====================

Introduction
------------

Local normalization is a module that normalizes the features of the local data (so that the p-norm ( `p-norm` ) of each feature after normalization takes a value of 1).

Specifically, for the feature matrix :math:`X`, perform `Local Normalization` transformation to obtain a new feature matrix :math:`\tilde{X}`. If features are normalized（`axis=0`）, then :math:`||\tilde{X}_{.j}||_p = 1\text{, }\forall j \in {1,\dots, m}`. If samples are normalized, then :math:`||\tilde{X}_{i.}||_p = 1\text{, }\forall i \in {1,\dots, n}`.

Parameter `train_info` defines different configuration information for each participant during the training process.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of the `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str``  Model name, should be `local_normalization`.
    - **config**: ``map`` Model configuration, `{}`, no need to config here.

**input**:
    - **trainset**: 
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder name of train dataset.
        - **name**: ``str`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**: 
        - **type**: ``str`` Validation dataset type, support `csv`.
        - **name**: ``str`` If type is `csv`, file name of validation dataset.
        - **path**: ``str`` If type is `csv`, folder name of validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
**output**:
    - **model**:
        - **type**: ``str`` Model output format, support "file".
        - **path**: ``str`` Folder name of output model.
        - **name**: ``str`` File name of output model.
    - **trainset**: 
        - **type**: ``str`` Output train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder name of output train dataset.
        - **name**: ``str`` If type is `csv`, file name of output train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**: 
        - **type**: ``str`` Output validation dataset type, support `csv`.
        - **name**: ``str`` If type is `csv`, file name of output validation dataset.
        - **path**: ``str`` If type is `csv`, folder name of output validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
        
**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **norm**: ``str`` Ways of normalization (``"l1"``/``"l2"``/``"max"``).
        - **axis**: ``int`` Dimension of normalization, 1 is row normalization, 0 is column normalization.
        - **featureNormalizeConfig**: ``map`` Different standardization settings for each column, of which the format is: {column name: {"norm": ``str``}, ..., column name: {"norm": ``str``}}.