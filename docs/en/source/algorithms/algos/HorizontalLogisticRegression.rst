==============================
Horizontal Logistic Regression
==============================

Introduction
------------

Horizontal logistic regression is a model obtained by building logistic regression model on the 
horizontal federated learning system.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of "label_trainer" or "assist trainer".

**model_info**:
    - **name**: ``str`` Model name, should be "horizontal_logistic_regression".
    - **config**:
        - **input_dim**: ``int`` Number of features of input data, can be Null. 
        - **bias**: ``bool`` Whether having a learnable additive bias.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset file type, such as "csv".
        - **path**: ``str`` Folder path of train dataset.
        - **name**: ``str`` File name of train dataset.
        - **has_label**: ``bool`` Whether dataset has label column.
        - **has_id**: ``bool`` Whether dataset has id column.
    - **valset**:
        - **type**: ``str`` Validation dataset file type, such as "csv".
        - **path**: ``str`` Folder path of Validation dataset.
        - **name**: ``str`` File name of Validation dataset.
        - **has_label**: ``bool`` Whether dataset has label column.
        - **has_id**: ``bool`` Whether dataset has id column.

**output**:
    - **model**: 
        - **type**: ``str`` Model output file format.
        - **path**: ``str`` Folder path of output model.
        - **name**: ``str`` File name of output model.
    - **metrics**:
        - **type**: ``str`` Metrics output file format.
        - **path**: ``str`` Folder path of output metrics.
        - **header**: ``bool`` Whether to include the column name.
    - **evaluation**:
        - **type**: ``str`` Evaluation output file format.
        - **path**: ``str`` Folder path of output Evaluation.
        - **header**: ``bool`` Whether to include the column name.

**train_info**:
    - **device**: ``str`` Device on which the algorithm runs, support "cpu".
    - **params**:
        - **global_epoch**: ``int`` Global training epoch.
        - **local_epoch**: ``int`` Local training epoch of involved parties.
        - **batch_size**: ``int`` Batch size of samples in local and global process. 
        - **aggregation_config**:
            - **type**: ``str`` Aggregation method, support "fedavg", "fedprox" and "scaffold".
            - **encryption**:
                - **method**: ``str`` Encryption method, recommend "otp".
                - **key_bitlength**: ``int`` Key length of one time pad encryption, support 64 and 128. 128 is recommended for better security.
                - **data_type**: ``str`` Input data type, support "torch.Tensor" and "numpy.ndarray", depending on model data type.
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **optimizer_config**: Support optimizers and their parameters defined in PyTorch or registered by user. For example:
            - **Adam**:
                - **lr**: ``float`` Optimizer learning rate.
                - **amsgrad**: ``bool`` Whether to use the AMSGrad variant.
        - **lr_scheduler_config**: Support lr_scheduler and their parameters defined in PyTorch or registered by user. For example:
            - **StepLR**:
                - **step_size**: ``int`` Period of learning rate decay.
                - **gamma**: ``float`` Multiplicative factor of learning rate decay.
        - **lossfunc_config**: Loss function configuration, support "BCEWithLogitsLoss".
        - **metric_config**: Support multiple metrics.
            - **accuracy**: Accuracy.
            - **precision**: Precision.
            - **recall**: Recall.
            - **f1_score**: F1 score.
            - **auc**: Area Under Curve.
            - **ks**: Kolmogorov-Smirnov (KS) Statistics.