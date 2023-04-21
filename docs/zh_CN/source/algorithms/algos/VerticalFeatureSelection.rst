===========================
Vertical Feature Selection
===========================

简介
----

在纵向联邦学习的框架上，根据iv值（:ref:`vertical-binning-woe-iv`）和皮尔逊相关系数（:ref:`vertical-pearson`）进行特征选择。
此算子由三个stage构成:

1. Vertical Binning Woe IV
2. Vertical Pearson
3. 根据以上两步结果，进行特征选择

参数列表
--------

前两个stage分别为vertical_binning_woe_iv和vertical_pearson, 具体参数解释见Vertical Binning Woe Iv和Vertical Pearson。
第三个stage为特征选择, 具体参数定义如下：

**identity**: ``str``  表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer)或 `trainer` (无标签的trainer)。

**model_info**:
    - **name**: ``str``  模型名称 (必须严格对应算法库中支持的模型), 在本模块中是 `vertical_pearson`。

**input**:
    - **trainset**:
        - **type**: ``str``  训练数据类型。
        - **path**: ``str``  训练数据所在文件读取路径。
        - **name**: ``str``  训练数据名。
        - **has_id**: ``bool`` 是否有id。
        - **has_label**: ``bool``  是否有label。
    - **valset**:
        - **type**: ``str``  验证数据类型。
        - **path**: ``str``  验证数据所在文件读取路径。
        - **name**: ``str``  验证数据名。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
    - **iv_result**:
        - **path**: ``str``  iv模型输出文件路径。
        - **name**: ``str``  iv模型输出文件名。
    - **corr_result**:
        - **path**: ``str``  pearson模型输出文件路径。
        - **name**: ``str``  pearson模型输出文件名。

**output**:
    - **path**: ``str``  模型输出文件路径。
    - **model**:
        - **name**: ``str`` 输出模型文件名。
    - **trainset**:
        - **name**: ``str``  经特征选择后的训练集相关输出文件名。
    - **valset**:
        - **name**: ``str``  经特征选择后的验证集相关输出文件名。

**train_info**:
    - **train_params**:
        - **filter**:
            - **common**:
                - **metrics**: ``str``  特征评估参数。仅支持 `iv`。
                - **filter_method**: ``str``  特征过滤方式。仅支持 `threshold`。
                - **threshold**: ``float``  过滤阈值。仅支持 `threshold`。
            - **correlation**:
                - **sort_metric**: ``str``  排序依据。仅支持 `iv`。
                - **correlation_threshold**: ``float``  相关性过滤阈值。