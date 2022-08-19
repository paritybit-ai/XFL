=======================
Local Data Split
=======================

Introduction
------------

Local data split is a module that split raw data into train and validation dataset.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_data_split`.
    - **config**: ``map`` Model configuration, no need to config here.

**input**:
    - **dataset**:
        - **type**: ``str`` Input dataset type, support `csv`.
        - **path**: ``str`` Folder path of input dataset.
        - **name**: ``str`` File name of input dataset. If None, all csv files under the folder path will be concated as the input dataset.
        - **has_label**: ``bool`` Whether dataset has label column.
        - **header**: ``bool`` Whether dataset has header. If True, the header of each input file must be the same.

**output**:
    - **trainset**: 
        - **path**: ``str`` Folder path of output train dataset.
        - **name**: ``str`` File name of output train dataset.
    - **valset**: 
        - **path**: ``str`` Folder path of output validation dataset.
        - **name**: ``str`` File name of output validation dataset.

**train_info**:  
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **params**:
        - **shuffle_params**: ``bool`` If True, input data will be shuffled.
        - **worker_num**: ``int`` Number of workers for parallel computing.
        - **batch_size**: ``int`` The size of small file in shuffle process.
        - **train_weight**: ``int`` The proportion of train dataset.
        - **val_weight**: ``int`` The proportion of validation dataset.
