=================
Vertical Sampler
=================

Introduction
------------

Two sample methods are provided: random sample and stratified sample. “Downsample” and “upsample” are supported in both methods.

Parameters List
---------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer` or `trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_sampler`.

**input**:  
    - **dataset**:
        - **type**: ``str`` Input dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
**output**:
    - **path**: ``str`` Folder path of output.
    - **sample_id**:
        - **name**: ``str`` File name of output model.
    - **dataset**:
        - **name**: ``bool`` File name of output trainset.

**train_info**:
    - **train_params**:
        - **method**: ``str`` Sample method, support "random" or "stratify".
        - **strategy**: ``str`` Sample strategy, support "downsample" or "upsample".
        - **random_seed**: ``int`` Random seed.
        - **fraction**: support three keys: "percentage" or "number" or "labeled_percentage".
            - **percentage**: ``float`` Threshold of "percentage" filter.
            - **number**: ``int`` Threshold of "number" filter.
            - **labeled_percentage**: ``list`` Threshold of "labeled_percentage" filter.
        - **marketing_specified**:
            - **threshold_method**: ``str`` Method of score filter, support "percentage" or "number" or "score".
            - **threshold**: ``int`` or ``float`` Threshold of filter.
