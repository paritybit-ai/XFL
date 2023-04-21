====================
Horizontal Bert
====================

Introduction
------------

Horizontal Bert model is a model obtained by building Bert proposed in the paper "BERT: 
Pre-training of Deep Bidirectional Transformers for Language Understanding" on the horizontal 
federation system and aims to solve sentiment analysis tasks in our scenario. "bert" is 
implemented based on TensorFlow framework. "bert_torch" is implemented based on PyTorch 
framework.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of "label_trainer" or "assist trainer".

**model_info**:
    - **name**: ``str`` Model name, should be "horizontal_bert" or "horizontal_bert_torch".
    - **config**:
        - **from_pretrained**: ``bool`` Whether to use pretrained model. Only support True.
        - **num_labels**: ``int`` Number of output labels.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset file type, such as "tsv".
        - **path**: ``str`` Folder path of train dataset.
        - **name**: ``str`` File name of train dataset.
    - **valset**:
        - **type**: ``str`` Validation dataset file type, such as "tsv".
        - **path**: ``str`` Folder path of Validation dataset.
        - **name**: ``str`` File name of Validation dataset.

**output**:
    - **model**: 
        - **type**: ``str`` Model output format, support "file".
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
    - **device**: ``str`` Device on which the algorithm runs, support "cpu" and specified gpu device such as `cuda:0`.
    - **interaction_params**
        - **save_frequency**: ``int`` Number of epoches of model saving interval.
        - **save_probabilities**: ``bool`` Whether to save the probability of model output.
        - **save_probabilities_bins_number**: ``int`` Number of bins of probability histogram.
        - **write_training_prediction**: ``bool`` Whether to save the prediction of training set.
        - **write_validation_prediction**: ``bool`` Whether to save the prediction of validation set.
        - **echo_training_metrics**: ``bool`` Whether to print the metrics of training set.
    - **params**:
        - **global_epoch**: ``int`` Global training epoch.
        - **local_epoch**: ``int`` Local training epoch of involved parties.
        - **batch_size**: ``int`` Batch size of samples in local and global process. 
        - **aggregation_config**:
            - **type**: ``str`` Aggregation method, support "fedavg", "fedprox" and "scaffold".
            - **encryption**:
                - **method**: ``str`` Encryption method, recommend "otp".
                - **key_bitlength**: ``int`` Key length of one time pad encryption, support 64 and 128. 128 is recommended for better security.
                - **data_type**: ``str`` Input data type, support "numpy.ndarray" for TensorFlow and "torch.Tensor" for PyTorch. 
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **optimizer_config**: Support optimizers and their parameters defined in Tensorflow, PyTorch or registered by user. For example:
            - **Adam**:
                - **lr**: ``float`` Learning rate.
                - **epsilon**: ``float`` Epsilon.
                - **clipnorm**: ``float`` Clipnorm.
        - **lr_scheduler_config**: Support lr_scheduler and their parameters defined in Tensorflow, PyTorch or registered by user. For example:
            - **CosinAnnealingLR**:
                - **T_max**: ``int`` Maximum iterations.
        - **lossfunc_config**: Support lossfunc and their parameters defined in Tensorflow, PyTorch or registered by user. For example:
            - **SparseCategoricalCrossentropy**:
        - **metric_config**: Support multiple metrics.
            - **accuracy**: Accuracy.
            - **precision**: Precision.
            - **recall**: Recall.
            - **f1_score**: F1 score.
            - **auc**: Area Under Curve.
            - **ks**: Kolmogorov-Smirnov (KS) Statistics.
        - **early_stopping**:
            - **key**: ``str`` Indicators of early stop strategy, such as "acc".
            - **patience**: ``int`` Tolerance number of early stop strategy.
            - **delta**: ``float`` Tolerance range of early stop strategy.