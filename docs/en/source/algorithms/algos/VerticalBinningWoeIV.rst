=========================
Vertical Binning Woe Iv
=========================

Introduction
------------

Vertical Binning Woe Iv is an algorithm for vertically calculating weight of evidence (WOE) and information value (IV).

Two types of binning method are supported:

1. Equal width binning: divides the data into :math:`k` intervals of equal size.
2. Equal frequency binning: divides the data into :math:`k` groups where each group contains approximately same number of values.

After binning, WOE and IV values can be calculated individually as follows(for :math:`i` th bin):

:math:`WOE_i = \ln \frac{y_i / y_T}{n_i/n_T}`

:math:`IV_i = \left( \frac{y_i}{y_T} - \frac{n_i}{n_T} \right) \times WOE_i`

where :math:`y_i` , :math:`n_i` denote the number of positive and negative samples of :math:`i` th bin respectively, 
:math:`y_T`, :math:`n_T` denote the number of positive and negative samples in total.

Parameters List
---------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer` or `trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_binning_woe_iv`.

**input**:
    - **trainset**: 
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``str`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
        - **nan_list**:  ``list`` List of special values, all and only the values in this list will be assigned to a single bin. 

**output**:
    - **path**: ``str`` Folder path of output.
    - **result**:
        - **name**: ``str`` File name of result.
    - **split_points**:
        - **name**: ``str`` File name of split points.

**train_info**:
    - **train_params**:
        - **encryption**: support two keys: "paillier" or "plain".
            - **paillier**:
                - **key_bit_size**: ``int`` Bit length of paillier key, recommended to greater than or equal to 2048.
                - **precision**: ``int`` Precison.
                - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
                - **parallelize_on**: ``bool`` Whether to use multicore for computing.
        - **binning**:
            - **method**: ``str`` Binning method, support "equal_width" or "equal_frequency".
            - **bins**: ``int`` Number of bins.
        - **max_num_cores**: ``int`` Number of cores for parallel computing.
