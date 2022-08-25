=================
Vertical Sampler
=================

Introduction
------------

Two sample methods are provided: random sample and stratified sample. “Downsample” and “upsample” are supported in both methods.

Parameters List
---------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `vertical_sampler`.
    - **config**: ``map`` Model configuration, no need to config here.

**input**:  
    - **dataset**:
        - **type**: ``str`` Input dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
**output**:
    - **model**:
        - **type**: ``str`` Type of output model.
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **trainset**:
        - **type**: ``str`` Output trainset type, support `csv`.
        - **path**: ``str`` Folder path of output trainset.
        - **name**: ``bool`` File name of output trainset.
        - **header**: ``bool`` Whether output trainset has id column.
        - **has_id**: ``bool`` Whether output trainset has id.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **method**: ``str`` Sample method, support "random" and "stratify".
        - **strategy**: ``str`` Sample strategy, support "downsample" and "upsample".
        - **random_state**: ``int`` Random seed.
        - **fraction**: ``int`` or ``float`` or ``str`` Sample fraction. When method == “stratify”, fraction should be sampling ratios of each category, e.g. "[(0,0.1), (1,0.2)]".
    - **infer_params**:
        - **threshold_method**: ``str`` Method of score filter, support "percentage", "number" and "score".
        - **threshold**: ``str`` Threshold of filter.
