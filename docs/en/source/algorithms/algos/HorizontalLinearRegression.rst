==============================
Horizontal Linear Regression
==============================

Introduction
------------

Horizontal linear regression is a model obtained by building linear regression model on the horizontal federated learning system.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.


**model_info**:
    - **name**: ``str`` Model name, should be `horizontal_linear_regression`.
    - **config**:
        - **input_dim**: ``int`` Size of each input sample. 
        - **bias**: ``int`` Whether having a learnable additive bias.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool``  If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.
    - **valset**:
        - **type**: ``str`` Validation dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of validation dataset.
        - **name**: ``bool`` If type is `csv`, file name of validation dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `csv`, whether dataset has label column.

**output**:  
    - **model**: 
        - **type**: ``str`` Model output format, support "file".
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support `cpu`.
    - **interaction_params**:
        - **save_frequency**: ``int`` Frequency (per global_epoch) to save model, set to -1 for not saving model.
    - **params**:
        - **global_epoch**: ``int`` Global training epoch.
        - **local_epoch**: ``int`` Local training epoch of involved parties.
        - **batch_size**: ``int`` Batch size of samples in local and global process. 
        - **aggregation_config**:
            - **type**: ``str`` Aggregation method, support "fedavg".
            - **encryption**:
                - **method**: ``str`` Encryption method, recommend "otp".
                - **key_bitlength**: ``int`` Key length of one time pad encryption，support 64 and 128.
                - **data_type**: ``str`` Input data type, support `torch.Tensor` and `numpy.ndarray`, depending on model data type.
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **optimizer_config**: Support optimizers and their parameters defined in pytorch or registered by user. For example:
            - **Adam**:
                - **lr**: ``float`` Optimizer learning rate.
                - **amsgrad**: ``bool`` Whether to use the AMSGrad variant.
        - **lr_scheduler_config**: Support lr_scheduler and their parameters defined in pytorch or registered by user. For example:
            - **StepLR**:
                - **step_size**: ``int`` Period of learning rate decay.
                - **gamma**: ``float`` Multiplicative factor of learning rate decay.
        - **lossfunc_config**: Loss function configuration, support `L1Loss` and `MAPEloss`.
        - **metric_config**: Support multiple metrics.
            - **mae**: Mean absolute error.
            - **mse**: Mean square error.
            - **mape**: Mean absolute percentage error.