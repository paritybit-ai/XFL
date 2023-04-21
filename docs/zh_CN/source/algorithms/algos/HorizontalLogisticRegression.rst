==============================
Horizontal Logistic Regression
==============================

简介
----

横向逻辑回归模型是将机器学习中的逻辑回归模型建立在横向联邦学习体系上得到的模型。

参数列表
--------

**identity**: ``str`` 表示该计算节点的身份类型, 只能为 "label_trainer" (带标签的 trainer) 或 "assit trainer" (无数据仅辅助计算的 trainer)。

**model_info**:
    - **name**: ``str`` 模型名称，需为 "horizontal_logistic_regression"。
    - **config**:
        - **input_dim**: ``int`` 输入数据的特征维度, 可为 Null。
        - **bias**: ``int`` 是否有截距。

**input**:
    - **trainset**:
        - **type**: ``str`` 训练数据集文件类型（后缀名），如 "csv"。
        - **path**: ``str`` 训练数据集所在文件夹路径。
        - **name**: ``str`` 训练数据集文件名。
        - **has_label**: ``bool`` 是否有 label。
        - **has_id**: ``bool`` 是否有 id。
    - **valset**:
        - **type**: ``str`` 验证数据集文件类型（后缀名），如 "csv"。
        - **path**: ``str`` 验证数据集所在文件夹路径。
        - **name**: ``str`` 验证数据集文件名。
        - **has_label**: ``bool`` 是否有 label。
        - **has_id**: ``bool`` 是否有 id。
**output**:
    - **model**:
        - **type**: ``str`` 模型输出文件类型（后缀名）。
        - **path**: ``str`` 模型输出文件夹路径。
        - **name**: ``str`` 模型输出文件名。
    - **metrics**:
        - **type**: ``str`` 指标输出文件类型（后缀名）。
        - **path**: ``str`` 指标输出文件夹路径。
        - **header**: ``bool`` 是否有列名。

**train_info**:
    - **device**: ``str`` 运行设备，目前只支持 “cpu” 。
    - **params**:
        - **global_epoch**: ``int`` 全局迭代次数。
        - **local_epoch**: ``int`` 本地迭代次数, 仅横向联邦框架需设置。
        - **batch_size**: ``int`` 每个批次训练样本的数量。
        - **aggregation_config**:
            - **type**: ``str`` 聚合方法，目前支持 "fedavg"，"fedprox" 和 "scaffold"。
            - **encryption**:
                - **method**: ``str`` 加密方法，推荐配置 "otp"。
                - **key_bitlength**: ``int`` 一次一密的密钥长度，支持64和128，推荐128，则安全强度为128。
                - **data_type**: ``str`` 输入的数据类型，目前支持 "torch.Tensor" 和 "numpy.ndarray"，根据模型实际使用的类型决定。
                - **key_exchange**:
                    - **key_bitlength**: ``int`` 密钥长度。
                    - **optimized**: ``bool`` 是否使用优化的方法。
                - **csprng**:
                    - **name**: ``str`` 伪随机数生成方法。
                    - **method**: ``str`` 该方法中使用的 Hash 算法。
        - **optimizer_config**: 支持 PyTorch 中定义的或用户自行注册的 optimizer，例如：
            - **Adam**:
                - **lr**: ``float`` 学习率。
                - **amsgrad**: ``bool`` 是否使用 amsgrad 优化算法。
        - **lr_scheduler_config**: 支持 PyTorch 中定义的或用户自行注册的 lr_scheduler，例如：
            - **StepLR**:
                - **step_size**: ``int`` 间隔单位。
                - **gamma**: ``float`` 学习率调整倍数。
        - **lossfunc_config**: 损失函数配置。暂时只支持 BCELoss。
        - **metric_config**:
            - **accuracy**: 准确率。
            - **precision**: 精确率。
            - **recall**: 召回率。
            - **f1_score**: 精确率和召回率的调和均值。
            - **auc**: roc 曲线下面积。
            - **ks**: 区分度评估指标。