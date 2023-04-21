======================================
Vertical XGBoost Distributed
======================================

Introduction
-----------------

The vertical XGBoost distributed model uses the Ray cluster for distributed computing, making it suitable for training models with large amounts of data.

Parameters List
-----------------

**Label_trainer**

**identity**: ``"label_trainer"``

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_xgboost_distributed`

**input**:
    - **trainset**:
        - **type**: ``str`` Training set type, supports `csv`.
        - **path**: ``str`` When `type` is `csv`, it means the path to the folder where the training set is located.
        - **name**: ``str`` or ``list`` Support single csv file or list of csv files when `type` is `csv`
        - **has_id**: ``bool`` When `type` is `csv`, it indicates whether there is an id column
        - **has_label**: ``bool`` When `type` is `csv`, it indicates whether there is a label column.
        - **missing_values**: ``list`` Indicates missing values.
        - **is_centralized**: ``bool`` Indicates whether the file is read uniformly from the ray_head node, currently only true is supported.
    - **valset**:
        - **type**: ``str`` Validation set type, supports `csv`
        - **path**: ``str`` When `type` is `csv`, it means the path to the folder where the validation set is located.
        - **name**: ``str`` or ``list`` Support single csv file or list of csv files when `type` is `csv`.
        - **has_id**: ``bool`` When `type` is `csv`, it indicates whether there is an id column.
        - **has_label**: ``bool`` When type is csv, it indicates whether there is a label column.
        - **missing_values**: ``list`` Indicates missing values.
        - **is_centralized**: ``bool`` Indicates whether the file is read uniformly from the rayhead node. Currently only true is supported.

**output**:
    - **path**: ``str`` Output directory path.
    - **model**:
        - **name**: ``str`` Model file name.
    - **metric_train**:
        - **name**: ``str`` Training set metric file name.
    - **metric_val**:
        - **name**: ``str`` Validation set metric file name
    - **prediction_train**:
        - **name**: ``str`` Training set prediction result file name
    - **prediction_val**:
        - **name**: ``str`` Validation set prediction result file name
    - **ks_plot_train**:
        - **name**: ``str`` Training set ks table file name
    - **ks_plot_val**:
        - **name**: ``str`` Validation set ks table file name
    - **decision_table_train**:
        - **name**: ``str`` Training set decision table file name
    - **decision_table_val**:
        - **name**: ``str`` Validation set decision table file name
    - **feature_importance**:
        - **name**: ``str`` Feature importance table file name

**train_info**:
    - **interaction_params**:
        - **save_frequency**: ``int`` Frequency of model saving in terms of trees, -1 means do not save intermediate models. 
        - **echo_training_metrics**: ``bool`` Whether to save training set metrics.
        - **write_training_prediction**: ``bool`` Whether to save training set prediction result.
        - **write_validation_prediction**: ``bool`` Whether to save validation set prediction result.

    - **train_params**:
        - **lossfunc**: ``map`` Loss function configuration. The format is: {loss function name: {specific configuration}}. For example: "BCEWithLogitsLoss": {}.
        - **num_trees**: ``int``  Number of trees.
        - **learning_rate**: ``float``  Learning rate.
        - **gamma**: ``float`` L1 regularization term for the number of leaf nodes.
        - **lambda**: ``float`` L2 regularization term for weights.
        - **max_depth**: ``int`` Maximum depth of the tree.
        - **num_bins**: ``int``  Number of bins.
        - **min_split_gain**: ``float`` Minimum split gain, positive.
        - **min_sample_split**: ``int`` Minimum number of samples in a tree node.
        - **feature_importance_type**: ``str``  Type of feature importance, supports gain and split.
        - **downsampling**: ``map``
            - **column**: ``map``
                - **rate**: ``float`` Feature dimension sampling rate.
            - **row**: ``map``
                - **run_goss**: ``bool`` Whether to use goss sampling.
                - **top_rate**: ``float`` The retain ratio of large gradient data in goss.
                - **other_rate**: ``float`` The retain ratio of small gradient data in goss, `0 < top_rate + other_rate <= 1`
        - **category**: ``map``
            - **cat_smooth**: ``float`` Parameter used to reduce the effect of noise on categorical features. The default value is 0.
            - **cat_feature**: ``map`` Configures the categorical features. The formula is: features that column indexes are in col_index if col_index_type is 'inclusive' or not in col_index if col_index_type is 'exclusive'. `union`` featuresthat column names are in col_names if col_names_type is 'inclusive' or not in col_names if col_names_type is 'exclusive'. `union if max_num_value_type is 'union' or intersect if max_num_value_type is 'intersection'` features that number of different values is less equal than max_num_value.
                - **col_index** ``str``: Index of the feature column that is (or is not) a categorical feature. Accepts slices or numbers, such as: "1, 4:5". The default value is "".
                - **col_names** ``list<str>``: Name of the feature column that is (or is not) a categorical feature. The default value is [].
                - **max_num_value** ``int``: If the number of unique values in a feature column is greater than or equal to this value, the feature column is a categorical feature. The default value is 0.
                - **col_index_type** ``str``: Supports 'inclusive' and 'exclusive'. The default value is 'inclusive'.
                - **col_names_type** ``str``: Supports 'inclusive' and 'exclusive'. The default value is 'inclusive'.
                - **max_num_value_type** ``str``: Supports 'intersection' and 'union'. The default value is 'union'.
        - **metric**: ``map`` Performance evaluation indicators. All the following key values are optional.
            - **decision_table**: ``map``
                - **method**: ``str`` Supports "equalfrequency" and "equalwith".
                - **bins**: ``int`` The number of divisions in the decision table.
            - **acc**: {}
            - **precision**: {}
            - **recall**: {}
            - **f1_score**: {}
            - **auc**: {}
            - **ks**: {}
        - **early_stopping**:
            - **key**: ``str`` The name of the metric used to judge whether to stop training. The metic name should have already been filled in the "metric" variable.
            - **patience**: ``int`` Number of steps with no improvement after which training will be stopped.
            - **delta**: ``float`` Minimum change in the value of metric to qualify as an improvement.
        - **encryption**:
            - **paillier**:
                - **key_bit_size**: ``int`` The bit length of Paillier key, which should be at least 2048 or more.
                - **precision**: ``int`` A precision-related parameter that can be null or a positive integer, such as 7.
                - **djn_on**: ``bool`` Whether to use the DJN method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multi-core parallel computing.
            - **plain**: ``map`` No encryption. Select either "plain" or "paillier".
        - **batch_size_val**: ``int`` The batch size for prediction on the validation set.
        - **atomic_row_size_per_cpu_core**: ``int`` The maximum number of rows per segment after the data is partitioned.
        - **pack_grad_hess**: ``bool`` Whether to pack the gradient and hessian into plaintext during encryption.


**Trainer**

**identity**: ``"trainer"``

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_xgboost_distributed`

**input**:
    - **trainset**:
        - **type**: ``str`` Training set type, supports `csv`.
        - **path**: ``str`` When `type` is `csv`, it means the path to the folder where the training set is located.
        - **name**: ``str`` or ``list`` Support single csv file or list of csv files when `type` is `csv`
        - **has_id**: ``bool`` When `type` is `csv`, it indicates whether there is an id column
        - **has_label**: ``bool`` When `type` is `csv`, it indicates whether there is a label column.
        - **missing_values**: ``list`` Indicates missing values.
        - **is_centralized**: ``bool`` Indicates whether the file is read uniformly from the ray_head node, currently only true is supported.
    - **valset**:
        - **type**: ``str`` Validation set type, supports `csv`
        - **path**: ``str`` When `type` is `csv`, it means the path to the folder where the validation set is located.
        - **name**: ``str`` or ``list`` Support single csv file or list of csv files when `type` is `csv`.
        - **has_id**: ``bool`` When `type` is `csv`, it indicates whether there is an id column.
        - **has_label**: ``bool`` When type is csv, it indicates whether there is a label column.
        - **missing_values**: ``list`` Indicates missing values.
        - **is_centralized**: ``bool`` Indicates whether the file is read uniformly from the rayhead node. Currently only true is supported.

**output**:
    - **path**: ``str`` Output directory path.
    - **model**:
        - **name**: ``str`` Model file name.

**train_info**:
    - **train_params**:
        - **downsampling**: ``map``
            - **column**: ``map``
                - **rate**: ``float`` Feature dimension sampling rate.
        - **category**: ``map``
            - **cat_feature**: ``map`` Configures the categorical features. The formula is: features that column indexes are in col_index if col_index_type is 'inclusive' or not in col_index if col_index_type is 'exclusive'. `union`` featuresthat column names are in col_names if col_names_type is 'inclusive' or not in col_names if col_names_type is 'exclusive'. `union if max_num_value_type is 'union' or intersect if max_num_value_type is 'intersection'` features that number of different values is less equal than max_num_value.
                - **col_index** ``str``: Index of the feature column that is (or is not) a categorical feature. Accepts slices or numbers, such as: "1, 4:5". The default value is "".
                - **col_names** ``list<str>``: Name of the feature column that is (or is not) a categorical feature. The default value is [].
                - **max_num_value** ``int``: If the number of unique values in a feature column is greater than or equal to this value, the feature column is a categorical feature. The default value is 0.
                - **col_index_type** ``str``: Supports 'inclusive' and 'exclusive'. The default value is 'inclusive'.
                - **col_names_type** ``str``: Supports 'inclusive' and 'exclusive'. The default value is 'inclusive'.
                - **max_num_value_type** ``str``: Supports 'intersection' and 'union'. The default value is 'union'.
        - **batch_blocks_on_recv**: ``int`` Number of data segments processed per batch on receive. 
        - **ray_col_step**: ``int`` Number of data columns processed simultaneously in a ray computing node. Automatically set by the algorithm if null.





