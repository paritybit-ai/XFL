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

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_pearson`.
    - **config**: ``map`` Model configuration, no need to config here.

**input**:  
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.

**output**:  
    - **model**: 
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **column_indexes**: ``list or int`` Column indexes involved in calculation. If it is -1, all columns participate in the calculation.
        - **column_names**: ``str`` Column names involved in calculation. the format is "name1, ..., nameN". If both name and index are provided, the union set of them will be applied.
        - **encryption_params**:
            - **paillier**:
                - **key_bit_size**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                - **precision**: ``int`` Precison.
                - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multicore for computing.
        - **max_num_core**: ``int`` Max number of cpu cores used for computing.
        - **sample_size**: ``int`` Row sampling size for speeding up pearson computation.