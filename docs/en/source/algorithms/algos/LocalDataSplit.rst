=======================
Local Data Split
=======================

Introduction
------------

Local data split is a module that split raw data into train and validation dataset.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be `label_trainer` or `trainer`.

**model_info**:  
    - **name**: ``str`` Model name, should be `local_data_split`.

**input**:
    - **dataset**:
        - **type**: ``str`` Input dataset type, support `csv`.
        - **path**: ``str`` Folder path of input dataset.
        - **name**: ``str`` File name of input dataset. If None, all csv files under the folder path will be concated as the input dataset.
        - **has_label**: ``bool`` Whether dataset has label column.
        - **has_header**: ``bool`` Whether dataset has header. If True, the header of each input file must be the same.

**output**:
    - **path**: ``str`` Folder path of output.
    - **trainset**: 
        - **name**: ``str`` File name of output train dataset.
    - **valset**: 
        - **name**: ``str`` File name of output validation dataset.

**train_info**:  
    - **train_params**:
        - **shuffle**: ``bool`` If True, input data will be shuffled.
        - **max_num_cores**: ``int`` Number of workers for parallel computing.
        - **batch_size**: ``int`` The size of small file in shuffle process.
        - **train_weight**: ``int`` The proportion of train dataset.
        - **val_weight**: ``int`` The proportion of validation dataset.
