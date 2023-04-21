=======================
Local Standard Scaler
=======================

简介
-----

对每一列特征进行归一化，即减去均值后除以标准差,使得归一化后的特征均值为0，方差为1，即:math:`\mathbb{E}(\tilde{X_{.j}}) = 0` ，:math:`Var (\tilde{X_{.j}}) = 1` 。参数`train_info`定义了每个参与方在训练过程中的不同配置信息。

参赛列表
--------

**identity**: ``str``  表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer) 或 `trainer` (无标签的trainer)。

**model_info**:
    - **name**: ``str``  表示模型名称 (必须严格对应算法库中支持的模型)。

**input**:  输入相关配置。
    - **trainset**:
        - **type**: ``str``  训练数据类型。
        - **path**: ``str``  训练数据所在文件读取路径。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
        - **name**: ``str``  文件名称。
    - **valset**:
        - **type**: ``str``  验证数据类型。
        - **path**: ``str``  验证数据所在文件读取路径。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
        - **name**: ``str`` 文件名称。

**output**:
    - **model**:
        - **type**: ``str``  训练模型输出格式。
        - **path**: ``str``  训练模型输出文件路径。
        - **name**: ``str``  训练模型输出文件名。
    - **trainset**:
        - **type**: ``str``  训练集数据输出类型。
        - **path**: ``str``  训练集数据输出文件所在路径。
        - **has_id**: ``bool``  是否有id。
        - **header**: ``bool``  是否有表头。
        - **name**: ``str``  文件名称。
    - **valset**:
        - **type**: ``str``  验证集数据输出类型。
        - **path**: ``str``  验证集数据输出文件所在路径。
        - **has_id**: ``bool``  是否有id。
        - **header**: ``bool``  是否有表头。
        - **name**: ``str``  文件名称。

**train_info**:
    - **device**: ``str``  目前只支持 `cpu`。
    - **params**:
        - **with_mean**: ``str``  是否提供了均值, 若无则置0。
        - **with_std**: ``int``  是否提供了标准差, 若无则置1。
        - **feature_standard**: ``map`` 对各列进行不同的标准化设置, 可为Null。其值的格式为：{列名: {"with_mean": ``bool``, "with_std": ``bool``}, ..., 列名: {"with_mean": ``bool``, "with_std": ``bool``}}。