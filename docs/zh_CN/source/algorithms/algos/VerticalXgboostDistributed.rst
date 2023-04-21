==================================
Vertical XGBoost Distributed
==================================

简介
-----

纵向xgboost分布式模型使用ray集群进行分布式计算，适合数据量较大情况下的模型训练。

参数列表
--------

**有Label方**

**identity**: ``"label_trainer"``

**model_info**:
    - **name**: ``str`` 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `vertical_xgboost_distributed`

**input**:
    - **trainset**:
        - **type**: ``str`` 训练集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示训练集所在文件夹路径
        - **name**: ``str`` or ``list`` 当 `type` 为 `csv` 时，支持单个csv文件或者csv文件列表
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
        - **missing_values**: ``list`` 表示缺失值
        - **is_centralized**: ``bool`` 表示文件是否统一从ray_head节点读取，目前只支持true

    - **valset**:
        - **type**: ``str`` 验证集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示验证集所在文件夹路径
        - **name**: ``str`` or ``list`` 当 `type` 为 `csv` 时，支持单个csv文件或者csv文件列表
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
        - **missing_values**: ``list`` 表示缺失值
        - **is_centralized**: ``bool`` 表示文件是否统一从ray_head节点读取，目前只支持true

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
        - **save_frequency**: ``int`` 模型存储频率，以树的棵树为单位，-1表示不保存中间模型。
        - **echo_training_metrics**: ``bool`` 是否保存训练集的指标
        - **write_training_prediction**: ``bool`` 是否保存训练集预测结果
        - **write_validation_prediction**: ``bool`` 是否保存验证集预测结果

    - **train_params**:
        - **lossfunc**: ``map`` 损失函数配置. 格式为：{ `损失函数名` : { `具体配置` }}. 例如："BCEWithLogitsLoss": {}
        - **num_trees**: ``int``  树的个数
        - **learning_rate**: ``float``  学习率
        - **gamma**: ``float`` 叶子节点个数的L1正则项
        - **lambda**: ``float`` 权重的L2正则项
        - **max_depth**: ``int`` 树的最大深度
        - **num_bins**: ``int``  分箱个数
        - **min_split_gain**: ``float`` 最小分裂收益，正数
        - **min_sample_split**: ``int``  树节点中的最少样本数
        - **feature_importance_type**: ``str``  特征重要性类型，支持 `gain` 和 `split`
        - **downsampling**: ``map``
            - **column**: ``map``
                - **rate**: ``float`` 特征维度采样率
            - **row**: ``map``
                - **run_goss**: ``bool`` 是否使用goss样本采样
                - **top_rate**: ``float`` 高权重的样本比例
                - **other_rate**: ``float`` 低权重的样本比例，`0 < top_rate + other_rate <= 1`
        - **category**: ``map``
            - **cat_smooth**: ``float`` 用于减少噪声对类别特征的影响的参数. 默认为0
            - **cat_feature**: ``map`` 配置类别特征的参数. 公式为: features that column indexes are in col_index if col_index_type is 'inclusive' or not in col_index if col_index_type is 'exclusive'. `union`` featuresthat column names are in col_names if col_names_type is 'inclusive' or not in col_names if col_names_type is 'exclusive'. `union if max_num_value_type is 'union' or intersect if max_num_value_type is 'intersection'` features that number of different values is less equal than max_num_value
                - **col_index** ``str``: 是（或不是）类别特征的特征列索引。接受切片或数字，如: `"1, 4:5"` . 默认为""
                - **col_names** ``list<str>``: 是（或不是）类别特征的特征列名. 默认为[]
                - **max_num_value** ``int``: 若一列特征的唯一值数量大于等于该值，则该列特征是类别特征. 默认为0
                - **col_index_type** ``str``: 支持 'inclusive' and 'exclusive'. 默认为 'inclusive'.
                - **col_names_type** ``str``: 支持 'inclusive' and 'exclusive'. 默认为 'inclusive'.
                - **max_num_value_type** ``str``: 支持 'intersection' and 'union'. 默认为 'union'.
        - **metric**: ``map`` 性能评估指标，以下所有键值均为可选项
            - **decision_table**: ``map``
                - **method**: ``str`` 支持 "equal_frequency" and "equal_with"
                - **bins**: ``int`` 决策表中的分箱数量
            - **acc**: {}
            - **precision**: {}
            - **recall**: {}
            - **f1_score**: {}
            - **auc**: {}
            - **ks**: {}
        - **early_stopping**:
            - **key**: ``str`` 判断训练是否早停的指标名，支持metric中填写的指标
            - **patience**: ``int`` 早停前可接受的指标没有发生改善的最大步长
            - **delta**: ``float`` 指标变化值，低于改值视为没有改善
        - **encryption**:
            - **paillier**:
                - **key_bit_size**: ``int`` paillier密码密钥长度，至少应大于等于2048
                - **precision**: ``int`` 精度相关参数，可为null或正整数，如7
                - **djn_on**: ``bool`` 是否采用DJN方法来生成密钥对
                - **parallelize_on**: ``bool`` 是否使用多核并行计算
            - **plain**: ``map`` 无加密，"plain"和"paillier"二选一
        - **batch_size_val**: ``int`` 验证集上做预测时的batch大小
        - **atomic_row_size_per_cpu_core**: ``int`` 数据被划分后每个片段的最大行数
        - **pack_grad_hess**: ``bool`` 在加密时，是否将gradient和hessian打包成一个明文


**无Label方**

**identity**: ``"trainer"``

**model_info**:
    - **name**: ``str`` 表示模型名称 (必须严格对应算法库中支持的模型)，在本模块中是 `vertical_xgboost_distributed`

**input**:
    - **trainset**:
        - **type**: ``str`` 训练集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示训练集所在文件夹路径
        - **name**: ``str`` or ``list`` 当 `type` 为 `csv` 时，支持单个csv文件或者csv文件列表
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
        - **missing_values**: ``list`` 表示缺失值
        - **is_centralized**: ``bool`` 表示文件是否统一从ray_head节点读取，目前只支持true

    - **valset**:
        - **type**: ``str`` 验证集类型，支持 `csv`
        - **path**: ``str`` 当 `type` 为 `csv` 时，表示验证集所在文件夹路径
        - **name**: ``str`` or ``list`` 当 `type` 为 `csv` 时，支持单个csv文件或者csv文件列表
        - **has_id**: ``bool`` 当 `type` 为 `csv` 时，表示是否有id列
        - **has_label**: ``bool`` 当 `type` 为 `csv` 时，表示是否有label列
        - **missing_values**: ``list`` 表示缺失值
        - **is_centralized**: ``bool`` 表示文件是否统一从ray_head节点读取，目前只支持true

**output**:
    - **path**: ``str`` 输出文件夹路径
    - **model**:
        - **name**: ``str`` 输出模型文件名

**train_info**:
    - **train_params**:
        - **downsampling**: ``map``
            - **column**: ``map``
                - **rate**: ``float`` 特征维度采样率
        - **category**: ``map``
            - **cat_feature**: ``map`` 配置类别特征的参数. 公式为: features that column indexes are in col_index if col_index_type is 'inclusive' or not in col_index if col_index_type is 'exclusive'. `union`` featuresthat column names are in col_names if col_names_type is 'inclusive' or not in col_names if col_names_type is 'exclusive'. `union if max_num_value_type is 'union' or intersect if max_num_value_type is 'intersection'` features that number of different values is less equal than max_num_value
                - **col_index** ``str``: 是（或不是）类别特征的特征列索引。接受切片或数字，如: `"1, 4:5"` . 默认为""
                - **col_names** ``list<str>``: 是（或不是）类别特征的特征列名. 默认为[]
                - **max_num_value** ``int``: 若一列特征的唯一值数量大于等于该值，则该列特征是类别特征. 默认为0
                - **col_index_type** ``str``: 支持 'inclusive' and 'exclusive'. 默认为 'inclusive'.
                - **col_names_type** ``str``: 支持 'inclusive' and 'exclusive'. 默认为 'inclusive'.
                - **max_num_value_type** ``str``: 支持 'intersection' and 'union'. 默认为 'union'.
        - **batch_blocks_on_recv**: ``int`` 接收时一次所处理的数据片段数量
        - **ray_col_step**: ``int`` 在ray计算节点中一次同时处理的数据列数量，当为null算法自动设置





