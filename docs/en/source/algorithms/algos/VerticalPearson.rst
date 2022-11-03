=================
Vertical Pearson
=================

Introduction
------------

This algorithm calculates the pearson correlation coefficient matrix for the data owned by all the parties. 
Pearson correlation coefficient is the ratio between the covariance of two variables and the product of their standard deviations.

:math:`\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}`


Parameter List
--------------

**identity**: ``str`` The role of each participant in federated learning, should be `label_trainer` or `trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_pearson`.

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

**train_info**:  
    - **train_params**:
        - **col_index**: ``list or int`` Column indexes involved in calculation. If it is -1, all columns participate in the calculation.
        - **col_names**: ``str`` Column names involved in calculation. the format is "name1, ..., nameN". If both name and index are provided, the union set of them will be applied.
        - **encryption**:
            - **paillier**:
                - **key_bit_size**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                - **precision**: ``int`` Precison.
                - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multicore for computing.
            - **plain**: ``map`` No encryption, an alternative to `otp` encryption, please set to `"plain": {}`.
        - **max_num_core**: ``int`` Max number of cpu cores used for computing.
        - **sample_size**: ``int`` Row sampling size for speeding up pearson computation.