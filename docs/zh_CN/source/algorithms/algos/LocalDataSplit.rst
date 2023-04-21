=======================
Local Data Split
=======================

简介
------------

`Local Data Split` 是把原始数据切分成训练集和数据集的模块。

参数列表
--------------

**identity**: ``str`` 表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer) 或 `trainer` (无标签的trainer)。

**model_info**:  
    - **name**: 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `local_data_split`。

**input**:
    - **dataset**:
        - **type**: ``str`` 输入数据类型，支持 `csv`。
        - **path**: ``str`` 输入数据所在文件夹读取路径。
        - **name**: ``str`` 输入数据文件名。
        - **has_label**: ``bool`` 是否有label。
        - **header**: ``bool`` 是否有表头。

**output**:
    - **path**: ``str`` 输出文件所在路径。
    - **trainset**:
        - **name**: ``str`` 训练集输出文件名称。
    - **valset**:
        - **name**: ``str`` 验证集输出文件名称。

**train_info**:  
    - **train_params**:
        - **shuffle**: ``bool`` 如果设置为True，将对输入数据进行shuffle。
        - **max_num_cores**: ``int`` 并行计算的并行数。
        - **batch_size**: ``int`` shuffle过程中每个小文件的行数。
        - **train_weight**: ``int`` 训练数据集的权重。
        - **val_weight**: ``int`` 验证数据集的权重。
