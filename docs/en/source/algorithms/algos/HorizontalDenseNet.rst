====================
Horizontal DenseNet
====================

Introduction
------------

Horizontal DenseNet model is a model obtained by building the model DenseNet proposed in the paper "Densely Connected Convolutional Networks" on the horizontal federation system, 
and is implemented based on the deep learning framework.

Parameter List
--------------

**identity**: ``str`` Federated identity of the party, should be one of "label_trainer" or "assist trainer".

**model_info**:
    - **name**: ``str`` Model name, should be "horizontal_densenet".
    - **config**:
        - **num_classes**: ``int`` Number of output classes.
        - **layers**: ``int`` DenseNet layers, support 121, 169, 201 and 264.

**input**:
    - **trainset**:
        - **type**: ``str`` Train dataset file type, such as "npz".
        - **path**: ``str`` Folder path of train dataset.
        - **name**: ``str`` File name of train dataset.
    - **valset**:
        - **type**: ``str`` Validation dataset file type, such as "npz".
        - **path**: ``str`` Folder path of Validation dataset.
        - **name**: ``str`` File name of Validation dataset.

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
                - **data_type**: ``str`` Input data type, support "torch.Tensor" and "numpy.ndarray", depending on model data type.
                - **key_exchange**:
                    - **key_bitlength**: ``int`` Bit length of paillier key, recommend to be greater than or equal to 2048.
                    - **optimized**: ``bool`` Whether to use optimized method.
                - **csprng**:
                    - **name**: ``str`` Pseudo-random number generation method.
                    - **method**: ``str`` Corresponding hash method.
        - **optimizer_config**: Support optimizers and their parameters defined in PyTorch or registered by user. For example:
            - **SGD**:
                - **lr**: ``float`` Learning rate.
                - **momentum**: ``float`` Momentum.
                - **weight_decay**: ``float`` Weight decay rate.
        - **lr_scheduler_config**: Support lr_scheduler and their parameters defined in PyTorch or registered by user. For example:
            - **CosinAnnealingLR**:
                - **T_max**: ``int`` Maximum iterations.
        - **lossfunc_config**: Support lossfunc and their parameters defined in PyTorch or registered by user. For example:
            - **CrossEntropyLoss**:
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