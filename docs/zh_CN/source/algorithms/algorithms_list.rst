================
算法类别详单
================

符号约定
-----------

机器学习中有很多术语，为了便于读者阅读各个算法的介绍，我们做出如下符号约定。

- :math:`X`：特征矩阵


.. math::

   X = \begin{pmatrix}
    x_{11} & x_{12} & \dots & x_{1m} \\
    x_{21} & x_{22} & \dots &  x_{2m} \\
    \vdots & \vdots & \dots &  \vdots \\
    v_{n1} & x_{n2} & \dots &  x_{nm} \\
    \end{pmatrix}


其中，每一行为一个样本，记为 :math:`x_i \quad (i=1, \dots, n)` 或者 :math:`X_{i.}` 。每一列为一个特征，可以记为 :math:`X_{.j} (j = 1, \dots , m)`

- :math:`Y`: 监督学习里的标签数据。

.. math::

   Y = \begin{pmatrix}
   y_1 \\
   y_2 \\
   \vdots \\
   y_n  \\
   \end{pmatrix}

对于回归问题，:math:`y_i \in \mathcal{R}`。对于分类问题， :math:`y_i \in [1, 2, \dots, K]` ，其中 :math:`K` 是类别总数。

- 训练集，验证集和测试集：对于机器学习来说，我们的数据通常可以分为训练集，验证集和测试集。我们使用上标train, val和test来区分它们。比如 :math:`X^{train}` ，:math:`X^{val}` ，:math:`X^{test}` 分别表示训练特征，验证特征和测试特征。



算法列表
----------------

.. csv-table::
   :header: "算法", "模块", "描述"

   ":doc:`Local Normalization <./algos/LocalNormalization>`", "local/normalization", "对数据进行归一化"
   ":doc:`Local Standard Scaler <./algos/LocalStandardScaler>`", "local/standard_scaler", "对数据进行标准化"
   ":doc:`Local Data Split <./algos/LocalDataSplit>`", "local/data_split", "将原始数据集切分成训练集和测试集"
   ":doc:`Local Feature Preprocess <./algos/LocalFeaturePreprocess>`", "local/feature_preprocess", "特征预处理"
   ":doc:`Local Data Statistic <./algos/LocalDataStatistic>`", "local/data_statistic", "特征统计"
   ":doc:`Horizontal Linear Regression <./algos/HorizontalLinearRegression>`", "horizontal/linear_regression", "两方、多方横向线性回归"
   ":doc:`Horizontal Logistic Regression <./algos/HorizontalLogisticRegression>`", "horizontal/logistic_regression", "两方、多方横向逻辑回归"
   ":doc:`Horizontal ResNet <./algos/HorizontalResNet>`", "horizontal/Resnet", "两方、多方横向ResNet"
   ":doc:`Horizontal DenseNet <./algos/HorizontalDenseNet>`", "horizontal/Densenet", "两方、多方横向DenseNet"
   ":doc:`Horizontal VGG <./algos/HorizontalVGG>`", "horizontal/Vgg", "两方、多方横向VGG"
   ":doc:`Horizontal Bert <./algos/HorizontalBert>`", "horizontal/Bert", "两方、多方横向Bert"
   ":doc:`Horizontal Poisson Regression <./algos/HorizontalPoissonRegression>`", "horizontal/poisson_regression", "两方、多方横向Poisson回归"
   ":doc:`Vertical Binning Woe IV <./algos/VerticalBinningWoeIV>`", "vertical/binning_woe_iv", "对特征进行woe和iv值计算，支持等频和等宽两种分箱策略"
   ":doc:`Vertical Pearson <./algos/VerticalPearson>`", "vertical/pearson", "两方、多方纵向Pearson相关系数"
   ":doc:`Vertical Feature Selection <./algos/VerticalFeatureSelection>`", "vertical/feature_selection", "两方、多方纵向特征选择"
   ":doc:`Vertical Linear Regression <./algos/VerticalLinearRegression>`", "vertical/linear_regression", "两方、多方纵向线性回归"
   ":doc:`Vertical Poisson Regression <./algos/VerticalPoissonRegression>`", "vertical/poisson_regression", "两方、多方纵向泊松回归"
   ":doc:`Vertical Logistic Regression <./algos/VerticalLogisticRegression>`", "vertical/logistic_regression", "两方、多方纵向逻辑回归"
   ":doc:`Vertical XGBoost <./algos/VerticalXgboost>`", "vertical/xgboost", "两方、多方纵向xgboost模块"
   ":doc:`Vertical XGBoostDistributed <./algos/VerticalXgboostDistributed>`", "vertical/xgboost_distributed", "分布式两方、多方纵向xgboost模块"
   ":doc:`Vertical Kmeans <./algos/VerticalKMeans>` ", "vertical/kmeans", "两方、多方纵向kmeans模块"
   ":doc:`Vertical Sampler <./algos/VerticalSampler>` ", "vertical/sampler", "两方、多方纵向采样模块"


