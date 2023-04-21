.. _vertical-pearson:

=================
Vertical Pearson
=================

简介
-----

在纵向联邦学习框架上，计算特征之间的皮尔逊相关系数，并留存相关性系数矩阵。
两个变量之间的皮尔逊相关系数定义为两个变量之间的协方差和标准差的商：

:math:`\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}`


参数列表
--------

**identity**: ``str``  表示该计算节点的身份类型，只能为 `label_trainer` (带标签的trainer)或 `trainer` (无标签的trainer)。

**model_info**:
    - **name**: ``str`` 模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `vertical_pearson`

**input**:
    - **trainset**:
        - **type**: ``str`` 数据集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示训练集所在文件夹路径
        - **name**: ``str`` 当 `type` 为 `csv` 时，表示训练集文件名
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列

**output**:
    - **path**: ``str`` 输出文件夹路径
    - **model**:
        - **name**: ``str`` 输出模型文件名

**train_info**:
    - **train_params**:
        - **col_index**: ``list or int`` 参加计算皮尔逊相关系数的列索引，若为-1，则所有列都参与计算
        - **col_names**: ``str`` 参加计算皮尔逊相关系数的列名，特征名用逗号分隔，若同时设置col_index和col_names，则取两者并集。若column_index设为-1，则此参数可为空
        - **encryption**:
            - **paillier**:
                - **key_bit_size**: ``int`` paillier密码密钥长度，至少应大于等于2048
                - **precision**: ``int`` 精度相关参数，可为null或正整数，如7
                - **djn_on**: ``bool`` 是否采用DJN方法来生成密钥对
                - **parallelize_on**: ``bool`` 是否使用多核并行计算
            - **plain**: ``map`` 无加密，设为 `"plain": {}` . plain"和"paillier"二选一
        - **max_num_core**: ``int`` 可用最大cpu核数
        - **sample_size**: ``int`` 行采样大小，用于加速pearson系数的计算