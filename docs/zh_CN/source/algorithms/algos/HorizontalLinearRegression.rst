==============================
Horizontal Linear Regression
==============================

简介
----

横向线性回归模型是将机器学习中的线性回归模型建立在横向联邦学习体系上得到的模型。

参数列表
--------

**identity**: ``str``  表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer), `trainer` (无标签的trainer), `assit trainer` (无数据仅辅助计算的trainer)之一

**model_info**:
    - **name**: ``str``  表示模型名称 (必须严格对应算法库中支持的模型)。
    - **config**:
        - **input_dim**: ``int`` 输入数据维度, 可为Null。
        - **bias**: ``int``  是否有截距。

**input**:
    - **trainset**:
        - **type**: ``str``  训练数据类型。
        - **path**: ``str``  训练数据所在文件读取路径。
        - **name**: ``bool``  是否有id。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
    - **valset**:
        - **type**: ``str``  验证数据类型。
        - **path**: ``str``  验证数据所在文件读取路径。
        - **name**: ``bool``  是否有id。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。

**output**:  输出文件的相关配置。
    - **model**: 训练模型输出相关配置。
        - **type**: ``str``  训练模型输出格式。
        - **path**: ``str``  训练模型输出文件路径。
        - **name**: ``str``  训练模型输出文件名。

**train_info**:
    - **device**: ``str``  目前只支持 `cpu` 。
    - **interaction_params**:
        - **save_frequency**: ``int`` 非 中间模型存储频率, 单位是globalEpoch。当设置为负值时, 表明本次模型训练不进行中间结果留存。
        - **save_probabilities**: ``bool``  概率分箱留存开关。留存路径为config中配置的checkpoints路径。
        - **save_probabilities_bins_number**: ``int``  分箱数量, 目前为等频分箱。
        - **write_training_prediction**: ``bool`` 是否留存训练集上的预测结果。
        - **echo_training_metrics**: ``bool`` 训练阶段metrics输出模块开关。
        - **write_validation_prediction**: ``bool`` 是否留存验证集上的预测结果。
    - **params**:
        - **global_epoch**: ``int`` 可选参数, 代表全局epoch大小。
        - **local_epoch**: ``int`` 可选参数, 本地epoch大小, 仅横向联邦框架需设置。
        - **batch_size**: ``int`` 可选参数, 每个batch中训练样本的数量。
        - **aggregation_config**:
            - **type**: ``str``  整合方法, 目前仅支持FedAvg。
            - **encryption**:
                - **method**: ``str``  加密方法，推荐配置"otp"。
                - **key_bitlength**: ``int``  一次一密的密钥长度，支持64和128，推荐128，则安全强度为128。
                - **data_type**: ``str``  输入的数据类型，目前支持"torch.Tensor"和"numpy.ndarray"，根据模型实际使用的类型决定。
                - **key_exchange**:
                    - **key_bitlength**: ``int``  密钥长度。
                    - **optimized**: ``bool``  是否使用优化的方法。
                - **csprng**:
                    - **name**: ``str``  伪随机数生成方法。
                    - **method**: ``str``  该方法中使用的Hash算法。
        - **optimizer_config**:
            - **Adam**:
                - **lr**: ``float``  学习率。
                - **amsgrad**: ``bool``  是否使用amsgrad优化算法。
        - **lr_scheduler_config**:
            - **StepLR**:
                - **step_size**: ``int``  间隔单位。
                - **gamma**: ``float``  学习率调整倍数。
        - **lossfunc_config**: 损失函数配置：目前支持L1Loss和MAPEloss。
        - **metric_config**:
            - **mae**: 平均绝对误差（Mean Absolute Error）。
            - **mse**: 均方误差（Mean Square Error）。
            - **mape**: 平均绝对百分比误差（Mean Absolute Percentage Error）。