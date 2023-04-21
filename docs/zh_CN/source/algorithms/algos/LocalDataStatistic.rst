=======================
Local Data Statistic
=======================

简介
------------

`Local Data Statistic` 是一个用于计算数据集的统计指标的模块，包括均值、最大值、最小值、中位数等。

参数列表
--------------

**identity**: ``str`` 表示该计算节点的身份类型，应为 `label_trainer` 或 `trainer`。

**model_info**:  
    - **name**: ``str`` 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `local_data_statistic`。

**input**:
    - **dataset**:
        - **type**: ``str`` 输入数据类型，支持 `csv`。
        - **path**: ``str`` 输入数据所在文件夹读取路径。
        - **name**: ``str`` 输入数据文件名。
        - **has_label**: ``bool`` 是否有label。
        - **has_id**: ``bool`` 是否有表头。

**output**:
    - **path**: ``str`` 输出文件所在路径。
    - **summary**:
        - **name**: ``str`` 输出文件名。

**train_info**:  
    - **train_params**:
        - **quantile**: ``list`` 待计算分位数，默认：[0.25, 0.5, 0.75].
