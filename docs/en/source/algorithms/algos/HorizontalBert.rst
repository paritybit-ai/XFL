====================
Horizontal Bert
====================

Introduction
------------

Horizontal Bert model is a model obtained by building Bert proposed in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" on the horizontal federation system and aims to
solve sentiment analysis tasks in our scenario. It is implemented based on TensorFlow framework.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of `label_trainer`, `trainer` or `assist trainer`.

**model_info**:
    - **name**: ``str`` Model name, should be `horizontal_bert`.
    - **config**:
        - **num_labels**: ``int`` Number of output labels.
        - **hidden_dropout_prob**: ``float`` Hidden dropout probability.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset type, support `tsv`.
        - **path**: ``str`` If type is `tsv`, folder path of train dataset.
        - **name**: ``bool`` If type is `tsv`, file name of train dataset.
        - **has_id**: ``bool`` If type is `tsv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `tsv`, whether dataset has label column.
    - **valset**:
        - **type**: ``str`` Validation dataset type, support `tsv`.
        - **path**: ``str`` If type is `tsv`, folder path of validation dataset.
        - **name**: ``bool`` If type is `tsv`, file name of validation dataset.
        - **has_id**: ``bool`` If type is `tsv`, whether dataset has id column.
        - **has_label**: ``bool`` If type is `tsv`, whether dataset has label column.

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
                - **learning_rate**: ``float`` Optimizer learning rate.
                - **epsilon**: ``float`` Optimizer epsilon.
                - **clipnorm**: ``float`` Optimizer clipnorm.

        - **lossfunc_config**: Loss function configuration, support `SparseCategoricalCrossentropy`.
        - **metric_config**: Support multiple metrics.
            - **accuracy**: Accuracy.
            - **precision**: Precision.
            - **recall**: Recall.
            - **f1_score**: F1 score.