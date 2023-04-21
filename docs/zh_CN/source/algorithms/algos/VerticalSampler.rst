=================
Vertical Sampler
=================

简介
------------

提供两种采样方法: 随机采样和分层采样。 两种方式均支持“降采样”和“过采样”。

参数列表
---------------

**identity**: ``str`` 表示该计算节点的身份类型，只能为 `label_trainer` (带标签的trainer)，`trainer` (无标签的trainer)

**model_info**:  
    - **name**: ``str`` 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `vertical_sampler`

**input**:  
    - **dataset**:
        - **type**: ``str`` 数据集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示训练集所在文件夹路径
        - **name**: ``str`` 当 `type` 为 `csv` 时，表示训练集文件名
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
**output**:
    - **path**: ``str`` 输出文件夹路径
    - **sample_id**:
        - **name**: ``str`` 样本id文件名
    - **dataset**:
        - **name**: ``bool`` 采样后数据集文件名

**train_info**:
    - **train_params**:
        - **method**: ``str`` 采样方法，支持 `random` 和 `stratify` 两种方式
        - **strategy**: ``str`` 采样策略，支持 `downsample` 和 `upsample`
        - **random_seed**: ``int`` 随机种子
        - **fraction**: 支持三种关键字: `percentage` 或 `number` 或 `labeled_percentage`。当method设置为 `stratify` 时，此参数应为每个类别数据对应的比例列表， 例如： [[0,0.1], [1,0.2]]
            - **percentage**: ``float`` 设置 `percentage` 过滤方式时的阈值
            - **number**: ``int`` 设置 `number` 过滤方式时的阈值
            - **labeled_percentage**: ``list`` 设置 `labeled_percentage` 过滤方式时的阈值
