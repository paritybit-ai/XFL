=======================
Local Feature Preprocess
=======================

Introduction
------------

Local feature preprocess is a module that preprocess feature locally.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_feature_preprocess`.
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
        - **type**: ``str`` Type of output model.
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **trainset**:
        - **type**: ``str`` Type of output trainset.
        - **path**: ``str`` Folder path of output trainset.
        - **name**: ``str`` File name of output trainset.
        - **header**: ``str`` Whether output trainset has header.
        - **has_id**: ``str`` Whether output trainset has id.
    - **valset**:
        - **type**: ``str`` Type of output valset.
        - **path**: ``str`` Folder path of output valset.
        - **name**: ``str`` File name of output valset.
        - **header**: ``str`` Whether output valset has header.
        - **has_id**: ``str`` Whether output valset has id.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **missing_params**:
            - **missing_values**: ``int`` or ``float`` or ``str`` or ``list`` The placeholder for the missing values.
            - **strategy**: ``str`` The imputation strategy.
            - **fill_value**: ``str`` or ``numerical value`` When strategy == “constant”, fill_value is used to replace all occurrences of missing_values.
            - **missing_feat_params**: ``map`` Fine-grained configuration for column preprocess(axis = 0). The format is defined as: {column name: {"missing_values": placeholder for the missing values, "strategy": imputation strategy, "fill_value": imputation value when strategy == “constant”}, ...}.
        - **outlier_params**:
            - **outlier_values**: ``int`` or ``float`` or ``str`` or ``list`` The placeholder for the outlier values.
            - **outlier_feat_params**: ``map`` Fine-grained configuration for column preprocess(axis = 0). The format is defined as: {column name: {"outlier_values": placeholder for the outlier values}, ...}.
        - **onehot_params**:
            - **feature_params**: ``map`` Fine-grained configuration for column preprocess(axis = 0). The format is defined as: {column name: {}, ..., column name: {}}.
