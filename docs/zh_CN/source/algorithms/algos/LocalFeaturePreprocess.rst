===============================
Local Feature Preprocess
===============================

简介
------------

`Local Feature Preprocess` 是对特征进行预处理的模块。

参数列表
--------------

**identity**: ``str`` 表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer) 或 `trainer` (无标签的trainer)。

**model_info**:  
    - **name**: 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `local_feature_preprocess`。
    - **config**: ``map`` 模型配置，在这里不需要配置。

**input**:
    - **trainset**:
        - **type**: ``str`` 训练数据类型。
        - **path**: ``str`` 训练数据所在文件夹读取路径。
        - **name**: ``str`` 训练数据名称。
        - **has_id**: ``bool`` 训练数据是否有id。
        - **has_label**: ``bool`` 训练数据是否有label。
    - **valset**:
        - **type**: ``str`` 验证数据类型。
        - **path**: ``str`` 验证数据所在文件夹读取路径。
        - **name**: ``str`` 验证数据名称。
        - **has_id**: ``bool`` 验证数据是否有id。
        - **has_label**: ``bool`` 验证数据是否有label。

**output**:
    - **path**: ``str`` 模型输出文件路径。
    - **model**:
        - **name**: ``str`` 模型输出文件名称。
    - **trainset**:
        - **name**: ``str`` 训练集输出文件名称。
    - **valset**:
        - **name**: ``str`` 验证集输出文件名称。

**train_info**:
    - **train_params**:
        - **missing**:
            - **missing_values**: ``int`` 或 ``float`` 或 ``str`` 或 ``list`` 缺失值的表征。
            - **strategy**: ``str`` 缺失值填充策略。
            - **fill_value**: ``str`` 或 ``numerical value`` 当strategy设置为“constant”时, 用fill_value填充所有缺失值。
            - **missing_features**: ``map`` 对指定列做缺失值填充(axis = 0)，格式如下: {列名: {"missing_values": ``int`` or ``float`` or ``str`` or ``list``, "strategy": ``str``, "fill_value": ``str`` or ``numerical value``}, ...}.
        - **outlier**:
            - **outlier_values**: ``int`` 或 ``float`` 或 ``str`` 或 ``list`` 异常值的表征。
            - **outlier_features**: ``map`` 对指定列做异常值替代(axis = 0)，格式如下: {列名: {"outlier_values": ``int`` or ``float`` or ``str`` or ``list``}, ...}.
        - **onehot**:
            - **onehot_features**: ``map`` 对指定列做onehot转换(axis = 0)，格式如下: {列名: {}, ..., 列名: {}}.
