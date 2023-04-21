.. _vertical-binning-woe-iv:

=========================
Vertical Binning Woe Iv
=========================

简介
----

纵向对特征进行分箱（binning, 可选等频或等距方式），计算特征的证据权重（Weight Of Evidence, WOE）和信息量（Information Value, IV）指标，并保存相关计算结果。

分箱方式有两种：

1. 等宽分箱：将特征的取值范围分为 :math:`k` 个等宽的区间，每个区间当作一个分箱。
2. 等频分箱：把特征的取值按照从小到大的顺序排列，并根据 :math:`k` 分位数（ :math:`k-quantile` ）进行分箱。

分箱后，对于第 :math:`i` 分箱，可分别计算WOE和IV值如下

:math:`WOE_i = \ln \frac{y_i / y_T}{n_i/n_T}`

:math:`IV_i = \left( \frac{y_i}{y_T} - \frac{n_i}{n_T} \right) \times WOE_i`

其中 :math:`y_i` ， :math:`y_T` 分别是分箱 :math:`i` 和总样本里的正样本数， :math:`n_i` ， :math:`n_T` 分别是分箱 :math:`i` 和总样本里的负样本数。

参数列表
---------

**identity**: ``str``  表示该计算节点的身份类型, 只能为 `label_trainer` (带标签的trainer), `trainer` (无标签的trainer)。

**model_info**:
    - **name**: ``str`` 模型名称 (必须严格对应算法库中支持的模型), 在本模块中是 `vertical_binning_woe_iv`。

**input**:  输入相关配置。
    - **trainset**:
        - **type**: ``str``  训练数据类型。
        - **path**: ``str``  训练数据所在文件读取路径。
        - **name**: ``str``  训练数据文件名称。
        - **has_id**: ``bool``  训练数据是否有id。
        - **has_label**: ``bool``  训练数据是否有label。
        - **nan_list**:  ``list``  缺失值标志列表, 对出现在此列表中的值单独分箱处理。

**output**:
    - **path**: ``str`` 模型输出文件路径。
    - **result**:
        - **name**: ``str`` 模型输出文件名。
    - **split_points**:
        - **name**: ``str`` 分箱点输出文件名。


**train_info**:
    - **train_params**:
        - **encryption**: 加密相关配置。支持两种加密方式，可为"paillier"或"plain"。
            - **paillier**:
                - **key_bit_size**: ``int``  密钥长度，2048的安全强度为112，3072的安全强度为128，推荐2048。
                - **precision**: ``int``  精度相关参数，也可设置为null，则算法自动设置，推荐7。
                - **djn_on**: ``bool``  是否采用优化的加密方式，影响加密性能，推荐true。
                - **parallelize_on**: ``bool``  是否使用多核并行，主要影响加解密性能，推荐true。
        - **binning**:
            - **method**: ``str``  指定要进行分箱的方式, 可选值为"equal_width"或"equal_frequency"。
            - **bins**: ``int``  指定要分箱的箱数。
        - **max_num_cores**: ``int`` 并行计算的进程池数量。