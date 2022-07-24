.. _vertical-pearson:

=================
Vertical Pearson
=================

Introduction
------------

The Pearson correlation coefficient between features is calculated, and the correlation coefficient matrix is retained on the vertical federated learning system.
Pearson correlation coefficient is the ratio between the covariance of two variables and the product of their standard deviations.

:math:`\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}`


Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of the `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_pearson`.
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
    - **params**:
        - **column_indexes**: ``list`` Column indexes that participates in the calculation of the Pearson correlation coefficient. If it is -1, all columns participate in the calculation.
        - **column_names**: ``str`` Column names that participates in the calculation of the Pearson correlation coefficient, the feature names are separated by commas. if column_indexes is set to -1, this parameter could be null.
        - **encryption_params**:
            - **paillier**:
                - **key_bit_size**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                - **precision**: ``int`` Precison.
                - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multicore for computing.
        - **max_num_core**: ``int`` Max number of cpu cores for computing.
        - **sample_size**: ``int`` Row sampling size for speeding up pearson computation.