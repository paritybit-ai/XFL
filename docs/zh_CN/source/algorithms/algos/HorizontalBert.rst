====================
Horizontal Bert
====================

简介
------------

横向 Bert 模型是在论文 "BERT: Pre-training of Deep Bidirectional Transformers for Language 
Understanding" 中提出的 Bert 模型建立在横向联邦学习体系上得到的模型，旨在解决我们场景中的情感分析任务。
"bert" 基于 TensorFlow 框架实现。"bert_torch" 基于 PyTorch 框架实现。

参数列表
--------------

**identity**: ``str`` 表示该计算节点的身份类型, 只能为 "label_trainer" (带标签的 trainer) 或 "assit trainer" (无数据仅辅助计算的 trainer)。

**model_info**:
    - **name**: ``str`` 模型名称，需为 "horizontal_bert" 或 "horizontal_bert_torch".
    - **config**:
        - **from_pretrained**: ``bool`` 是否使用预训练模型。仅支持 True。
        - **num_labels**: ``int`` 模型输出标签类型数目.

**input**:
    - **trainset**:
        - **type**: ``str`` 训练数据集文件类型（后缀名），如 "tsv"。
        - **path**: ``str`` 训练数据集所在文件夹路径。
        - **name**: ``str`` 训练数据集文件名。
    - **valset**:
        - **type**: ``str`` 验证数据集文件类型（后缀名），如 "tsv"。
        - **path**: ``str`` 验证数据集所在文件夹路径。
        - **name**: ``str`` 验证数据集文件名。
**output**:
    - **model**:
        - **type**: ``str`` 模型输出文件类型（后缀名）。
        - **path**: ``str`` 模型输出文件夹路径。
        - **name**: ``str`` 模型输出文件名。
    - **metrics**:
        - **type**: ``str`` 指标输出文件类型（后缀名）。
        - **path**: ``str`` 指标输出文件夹路径。
        - **header**: ``bool`` 是否有列名。
    - **evaluation**:
        - **type**: ``str`` 验证集指标输出文件类型（后缀名）。
        - **path**: ``str`` 验证集指标输出文件夹路径。
        - **header**: ``bool`` 是否有列名。

**train_info**:
    - **device**: ``str`` 运行设备，支持 “cpu” 或 gpu 名例如 "cuda:0" 。
    - **interaction_params**
        - **save_frequency**: ``int`` 模型保存间隔的迭代次数。
        - **save_probabilities**: ``bool`` 是否保存模型输出的概率。
        - **save_probabilities_bins_number**: ``int`` 模型输出概率的分箱数目。
        - **write_training_prediction**: ``bool`` 是否保存训练集预测结果。
        - **write_validation_prediction**: ``bool`` 是否保存验证集预测结果。
        - **echo_training_metrics**: ``bool`` 是否在训练过程中输出训练集指标。
    - **params**:
        - **global_epoch**: ``int`` 全局迭代次数。
        - **local_epoch**: ``int`` 本地迭代次数, 仅横向联邦框架需设置。
        - **batch_size**: ``int`` 每个批次训练样本的数量。
        - **aggregation_config**:
            - **type**: ``str`` 聚合方法，目前支持 "fedavg"，"fedprox" 和 "scaffold"。
            - **encryption**:
                - **method**: ``str`` 加密方法，推荐配置 "otp"。
                - **key_bitlength**: ``int`` 一次一密的密钥长度，支持64和128，推荐128，则安全强度为128。
                - **data_type**: ``str`` 输入的数据类型，TensorFlow 支持 "numpy.ndarray"，PyTorch 支持 "torch.Tensor"。
                - **key_exchange**:
                    - **key_bitlength**: ``int`` 密钥长度。
                    - **optimized**: ``bool`` 是否使用优化的方法。
                - **csprng**:
                    - **name**: ``str`` 伪随机数生成方法。
                    - **method**: ``str`` 该方法中使用的 Hash 算法。
        - **optimizer_config**: 支持 Tensorflow、PyTorch 中定义的或用户自行注册的 optimizer，例如：
            - **Adam**:
                - **lr**: ``float`` 学习率。
                - **epsilon**: ``float`` Epsilon.
                - **clipnorm**: ``float`` 梯度裁剪的范数。
        - **lr_scheduler_config**: 支持 Tensorflow、PyTorch 中定义的或用户自行注册的 lr_scheduler，例如：
            - **CosinAnnealingLR**:
                - **T_max**: ``int`` 最大迭代次数。
        - **lossfunc_config**: 支持 Tensorflow、PyTorch 中定义的或用户自行注册的 lossfunc，例如：
            - **SparseCategoricalCrossentropy**:
        - **metric_config**:
            - **accuracy**: 准确率。
            - **precision**: 精确率。
            - **recall**: 召回率。
            - **f1_score**: 精确率和召回率的调和均值。
            - **auc**: roc 曲线下面积。
            - **ks**: 区分度评估指标。
        - **early_stopping**:
            - **key**: ``str`` 早停策略的指标，例如 "acc"。
            - **patience**: ``int`` 早停策略的容忍次数。
            - **delta**: ``float`` 早停策略的容忍范围。