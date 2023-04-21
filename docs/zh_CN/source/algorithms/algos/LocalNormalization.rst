====================
Local Normalization
====================

简介
-----

`Local Normalization` 是对本地数据进行特征的归一化（使得归一化后每个特征的p-范数（ `p-norm` ）取值为一）的模块。

具体来说，对于特征矩阵 :math:`X` 进行 `Local Normalization` 变换得到新的特征矩阵 :math:`\tilde{X}` ，如果对特征进行归一化（`axis=0`），那么 :math:`||\tilde{X}_{.j}||_p = 1\text{, }\forall j \in {1,\dots, m}` 。对样本进行归一化，则有 :math:`||\tilde{X}_{i.}||_p = 1\text{, }\forall i \in {1,\dots, n}` 。

接收的参数 `train_info` 定义了各参与方在训练中的配置信息，须在各参与方中保持一致。

参数列表
--------


**identity**: ``str``  表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer) 或 `trainer` (无标签的trainer)。

**model_info**:
    - **name**: ``str``  表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `local_normalization`。

**input**:  输入相关配置。
    - **trainset**:
        - **type**: ``str``  训练数据类型。
        - **path**: ``str``  训练数据所在文件夹读取路径。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
        - **name**: ``str``  文件名称。
    - **valset**:
        - **type**: ``str``  验证数据类型。
        - **path**: ``str``,  验证数据所在文件夹读取路径。
        - **has_id**: ``bool``  是否有id。
        - **has_label**: ``bool``  是否有label。
        - **name**: ``str``  文件名称。

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
        - **norm**: ``str``  归一化方式 (``"l1"``/``"l2"``/``"max"``)。
        - **axis**: ``int``  归一化维度, 1为行归一化, 0为列归一化。
        - **feature_norm**: ``map`` 对各列进行不同的归一化设置, 可为Null。其值的格式为：{列名: {"norm": ``str``}, ..., 列名: {"norm": ``str``}}。