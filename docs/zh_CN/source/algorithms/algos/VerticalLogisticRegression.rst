=============================
Vertical Logistic Regression
=============================

简介
-----

纵向逻辑回归模型是将机器学习中的逻辑回归模型建立在纵向联邦体系上得到的模型。

参数列表
--------

**identity**: ``str``  表示该计算节点的身份类型，只能为 `label_trainer` (带标签的trainer)，`trainer` (无标签的trainer)，`assit trainer` (无数据仅辅助计算的trainer)之一

**model_info**:
    - **name**: ``str``  表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `vertical_logistic_regression`

**input**:
    - **trainset**:
        - **type**: ``str`` 数据集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示训练集所在文件夹路径
        - **name**: ``str`` 当 `type` 为 `csv` 时，表示训练集文件名
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
    - **valset**:
        - **type**: ``str`` 验证集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示验证集所在文件夹路径
        - **name**: ``str`` 当 `type` 为 `csv` 时，表示验证集文件名
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
    - **pretrained_model**: ``map``
        - **path**: ``str`` 预训练模型文件夹路径
        - **name**: ``str`` 预训练模型名

**output**:
    - **path**: ``str`` 输出文件夹路径
    - **model**:
        - **name**: ``str`` 输出模型文件名
    - **metric_train**:
        - **name**: ``str`` 训练集指标文件名
    - **metric_val**:
        - **name**: ``str`` 验证集指标文件名
    - **prediction_train**:
        - **name**: ``str`` 训练集预测结果文件名
    - **prediction_val**:
        - **name**: ``str`` 验证集预测结果文件名
    - **ks_plot_train**:
        - **name**: ``str`` 训练集ks表文件名
    - **ks_plot_val**:
        - **name**: ``str`` 验证集ks表文件名
    - **decision_table_train**:
        - **name**: ``str`` 训练集决策表文件名
    - **decision_table_val**:
        - **name**: ``str`` 验证集决策表文件名
    - **feature_importance**:
        - **name**: ``str`` 特征重要性表文件名

**train_info**:
    - **interaction_params**:
        - **save_frequency**: ``int`` 模型存储频率，-1表示不保存中间模型
        - **echo_training_metrics**: ``bool`` 是否保存训练集的指标
        - **write_training_prediction**: ``bool`` 是否保存训练集预测结果
        - **write_validation_prediction**: ``bool`` 是否保存验证集预测结果

    - **train_params**:
        - **global_epoch**: ``int`` 训练轮数
        - **batch_size**: ``int`` 训练batch大小
        - **encryption**: ``map`` 支持"ckks" 或者 "paillier".
            - **ckks**: ``map``
                - **poly_modulus_degree**: ``int``  多项式模次数
                - **coeff_mod_bit_sizes**: ``list``  系数模位数
                - **global_scale_bit_size**: ``int`` 全局缩放因子位数
            - **paillier**:
                - **key_bit_size**: ``int`` paillier密码密钥长度，至少应大于等于2048
                - **precision**: ``int`` 精度相关参数，可为null或正整数,如7
                - **djn_on**: ``bool`` 是否采用DJN方法来生成密钥对
                - **parallelize_on**: ``bool`` 是否使用多核并行计算
        - **optimizer**: 
            - **lr**: ``float``  学习率
            - **p**: ``int``  正则化参数，"0"/"1"/"2"分别代表不加正则/l1正则/l2正则
            - **alpha**: ``float``  惩罚系数

        - **metric**: ``map`` 性能评估指标，以下所有键值均为可选项
            - **decision_table**: ``map``
                - **method**: ``str`` 支持 "equal_frequency" 和 "equal_with"。
                - **bins**: ``int`` 决策表中的分箱数量。
            - **acc**: ``map`` 准确率, 仅支持{}。
            - **precision**: ``map`` 精确率, 仅支持{}。
            - **recall**: ``map`` 召回率, 仅支持{}。
            - **f1_score**: ``map`` 精确率和召回率的调和平均值, 仅支持{}。
            - **auc**: ``map`` 曲线下面积, 仅支持{}。
            - **ks**: ``map`` ks曲线, 仅支持{}。

        - **early_stopping**:
            - **key**: ``str`` 判断训练是否早停的指标名，支持metric中填写的指标。
            - **patience**: ``int`` 早停前可接受的指标没有发生改善的最大步长。
            - **delta**: ``float`` 指标变化值，低于改值视为没有改善。

        - **random_seed**: ``int`` 用于打乱数据集的随机种子，可为None。


.. [Yang2019] Yang S, Ren B, Zhou X, et al. Parallel distributed logistic regression for vertical federated learning without third-party coordinator[J]. arXiv preprint arXiv:1911.09824, 2019.