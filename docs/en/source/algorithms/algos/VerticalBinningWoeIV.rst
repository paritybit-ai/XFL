=========================
Vertical Binning Woe Iv
=========================

Introduction
------------

Vertical Binning Woe Iv is an algorithm for vertical feature binning and calculation of the weight of evidence (WOE) and information value (IV).

There are two ways of binning:

1. Equal width binning: divides the data into :math:`k` intervals of equal size.
2. Equal frequency binning: divides the data into :math:`k` groups which each group contains approximately same number of values.

After binning, for the :math:`i` bin, WOE and IV values can be calculated separately as follows:

:math:`WOE_i = \ln \frac{y_i / y_T}{n_i/n_T}`

:math:`IV_i = \left( \frac{y_i}{y_T} - \frac{n_i}{n_T} \right) \times WOE_i`

where :math:`y_i` , :math:`n_i` are respectively the number of positive and negative samples of bin :math:`i`, :math:`y_T`, :math:`n_T` are respectively the number of positive and negative samples in total.

Parameters List
---------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `vertical_binning_woe_iv`.
    - **config**: ``map`` Model configuration, no need here.

**input**:
    - **trainset**: 
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``str`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
        - **missing_values**:
            - **has_missing**: ``bool`` Whether having missing values.
            - **format**: ``str`` Missing value format, default: "null".
            - **strategy**: ``str`` Filling method, default: "mean".
            - **fulfill_value**: ``int`` Filling value if strategy is `constant`.
        - **nan_list**:  ``list`` List of missing value flags, values in this list will be binned individually.

**output**:
    - **trainset**:
        - **path**: ``str`` Folder path of output dataset.
        - **name**: ``str`` File name of output dataset.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **encryption_params**:
            - **method**: ``str`` Encryption method, recommend "paillier".
            - **key_bit_size**: ``int`` Bit length of paillier key, recommended to greater than or equal to 2048.
            - **precision**: ``int`` Precison.
            - **djn_on**: ``bool`` Whether to use djn method to generate key pair.
            - **parallelize_on**: ``bool`` Whether to use multicore for computing.
        - **binning_params**:
            - **method**: ``str`` Binning way, "equalWidth" or "equalFrequency".
            - **bins**: ``int`` Number of bins.
        - **woe_iv_params**: ``repeated<map>`` Different binning settings for each column, of which the format is: {column name: {method: `str`, bins: `int`},..., column name: {method: `str`, bins: `int`}}.
        - **pool_num**: ``int`` Number of process pools for parallel computing.