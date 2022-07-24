=================
Vertical K-means
=================

Introduction
------------

Vertical Kmeans is a model obtained by building Kmeans model in machine learning on the vertical federated learning system.
The calculation process for vertical Kmeans (two parties in this example) is shown as follows, where A is a labeled participant, B is an unlabeled participant, and C is an assist party that performs aggregation operations.

.. image:: ../../images/vertical_kmeans_en.png

Parameters List
---------------

**identity**: ``str`` Federated identity of the party, should be one of the `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_kmeans`.
    - **config**: ``map`` Model configuration, `{}`, no need to config here.

**input**:  
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder name of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
**output**:  
    - **model**: 
        - **type**: ``str`` Model output format, support "file".
        - **path**: ``str`` Folder name of output model.
        - **name**: ``str`` File name of output model.


**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **aggregation_config**:
        - **type**: ``str`` Aggregation method, support "fedavg"。
        - **encryption**:
            - **method**: ``str`` Encryption, recommend "otp".
            - **key_bitlength**: ``int`` Key length of one time pad encryption, support 64 and 128.
            - **data_type**: ``str`` Input data type, support `torch.Tensor` and `numpy.ndarray`, depending on model data type。
            - **key_exchange**:
                - **key_bitlength**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                - **optimized**: ``bool`` Whether to use optimized method.
            - **csprng**:
                - **name**: ``str`` Pseudo-random number generation method.
                - **method**: ``str`` Corresponding hash method.
            - **weight_factor**: ``list`` or ``float`` Weight factor. Each non-scheduler node needs to set weight_factor representing the weight of the model parameters of this node, while the scheduler does not need to be set.
            - **params**:
                - **k**: ``int`` Number of clusters.
                - **max_iter**: ``int`` Maximum iteration.
                - **tol**: ``float`` Convergence threshold.
                - **extra_config**:
                    - **shuffle_seed**: ``int`` Random seed.