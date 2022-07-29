====================
Horizontal ResNet
====================

Introduction
------------

Horizontal ResNet model is a model obtained by building the classic model ResNet proposed in the paper "Deep Residual Learning for Image Recognition" on the horizontal federation system, 
and is implemented based on the deep learning framework.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `horizontal_resnet`.
    - **config**:
        - **num_classes**: ``int`` Number of output classes.
        - **layers**: ``int`` ResNet layers(50,101,152), support ResNet50, ResNet101, ResNet152.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `csv`.
        - **path**: ``str`` If type is `csv`, folder path of train dataset.
        - **name**: ``bool`` If type is `csv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `csv`, whether dataset has id column.
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
    - **device**: ``str`` Device on which the algorithm runs, support `cpu` and specified gpu device such as `cuda:0`.
    - **params**:
        - **global_epoch**: ``int`` Global training epoch.
        - **local_epoch**: ``int`` Local training epoch of involved parties.
        - **batch_size**: ``int`` Batch size of samples in local and global process. 
        - **aggregation_config**:
            - **type**: ``str`` Aggregation method, support "fedavg".
            - **encryption**:
                - **method**: ``str`` Encryption method, recommend "otp".
                - **key_bitlength**: ``int`` Key length of one time pad encryption, support 64 and 128.
                - **data_type**: ``str`` Input data type, support `torch.Tensor` and `numpy.ndarray`, depending on model data type.
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bitlength of paillier key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **optimizer_config**: Support optimizers and their parameters defined in pytorch or registered by user. For example:
            - **Adam**:
                - **lr**: ``float`` Optimizer earning rate.
                - **amsgrad**: ``bool`` Whether to use the AMSGrad variant.
        - **lr_scheduler_config**: Support lr_scheduler and their parameters defined in pytorch or registered by user. For example:
            - **StepLR**:
                - **step_size**: ``int`` Period of learning rate decay.
                - **gamma**: ``float`` Multiplicative factor of learning rate decay.
        - **lossfunc_config**: Loss function configuration, support `CrossEntropyLoss`.
        - **metric_config**: Support multiple metrics.
            - **accuracy**: Accuracy.
            - **precision**: Precision.
            - **recall**: Recall.
            - **f1_score**: F1 score.