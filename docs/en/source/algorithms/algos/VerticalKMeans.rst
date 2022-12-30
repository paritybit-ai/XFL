=================
Vertical K-means
=================

Introduction
------------

The calculation process for vertical Kmeans (two parties in this example) is illustracted as follows, 
where A and B are the trainer and the label_trainer who own the data, C is the assist_trainer who performs aggregation operations.

.. image:: ../../images/vertical_kmeans_en.png

Parameters List
---------------

**identity**: ``str`` The role of each participant in federated learning, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_kmeans`.

**computing_engine**: ``str`` The computing engine to run the algorithm, currently supported are `local` or `spark`.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, currently supported is `csv`.
        - **path**: ``str`` The folder path of train dataset.
        - **name**: ``str`` The file name of train dataset.
        - **has_id**: ``bool`` Whether the dataset has id column.
        - **has_label**: ``bool`` Whether the dataset has label column.
**output**:
    - **path**: ``str`` Output folder path.
    - **model**:
        - **name**: ``str`` File name of output model.
    - **result**:
        - **name**: ``str`` File name of result.
    - **summary**:
        - **name**: ``str`` File name of summary.

**train_info**:  
    - **train_params**: ``map``
        - **encryption**:
            - **otp**:
                - **key_bitlength**: ``int`` Key length of one time pad encryption, support 64 and 128.
                - **data_type**: ``str`` Input data type, support `torch.Tensor` and `numpy.ndarray`, depending on model data type.
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bit length of the key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **k**: ``int`` Number of clusters.
        - **max_iter**: ``int`` Maximum iteration.
        - **tol**: ``float`` Convergence threshold.
        - **random_seed**: ``int`` Random seed.
