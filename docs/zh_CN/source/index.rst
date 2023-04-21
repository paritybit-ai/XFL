.. XFL documentation master file, created by
   sphinx-quickstart on Thu Jul  7 17:24:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用XFL文档!
===============================

XFL是一个高性能、高灵活度、高适用性、轻量、开放易用的联邦学习框架。支持横向、纵向联邦场景下的多种联邦模型，
为了使得用户在合法合规的基础上进行联合建模、挖掘数据的价值，XFL采用同态加密、差分隐私、安全多方计算等多种安全技术保
护用户本地数据不被泄露，并且使用安全通信协议来确保通信安全。


特点：
  - 高性能的算法库

    - 算法全面：支持多种主流横向/纵向联邦算法
    - 性能优异：性能大幅领先于信通院性能评测平均水平
    - 网络优化：在弱网络、高延迟、大量丢包、较长时间断网的情况下依然可以完成训练任务

  - 应用部署灵活

    - 参与方： 支持两方/多方联邦学习
    - 任务调度： 无论有无标签，任何一方都可以作为任务发起方
    - 硬件支持： 支持CPU/GPU/混合部署

  - 轻量开放易用

    - 轻量： 对服务器性能要求低，部分算法可在性能较差的环境下运行
    - 开放：支持 Pytorch，Tensorflow，PaddlePaddle 和 Jax 等主流机器学习框架，支持用户自定义横向模型


功能支持
----------

====================    ==============
功能                      是否支持
====================    ==============
横向联邦                       ✅
纵向联邦                       ✅
XGBoost                      ✅
深度学习框架              Pytorch/Tensorflow/PaddlePaddle/Jax
半同态加密                     ✅
全同态加密                     ✅
一次一密                       ✅
多方安全计算                   ✅
差分隐私                       ✅
安全对齐                       ✅
隐匿查询                       ✅
GPU支持                        ✅
集群部署                       ✅
在线推理                       ✅
联邦节点管理                    ✅
联邦数据管理                    ✅
====================    ==============


.. toctree::
   :maxdepth: 2
   :caption: 教程

   简介 <tutorial/introduction.md>
   使用教程 <tutorial/usage.md>

.. toctree::
   :maxdepth: 2
   :caption: 算法

   算法列表 <algorithms/algorithms_list.rst>
   密码算法 <algorithms/cryptographic_algorithm.rst>
   ./algorithms/aggregation_algorithm
   ./algorithms/differential_privacy
   ./algorithms/algos/HorizontalLinearRegression
   ./algorithms/algos/HorizontalLogisticRegression
   ./algorithms/algos/HorizontalResNet
   ./algorithms/algos/HorizontalDenseNet
   ./algorithms/algos/HorizontalVGG
   ./algorithms/algos/HorizontalBert
   ./algorithms/algos/HorizontalPoissonRegression
   ./algorithms/algos/VerticalLogisticRegression
   ./algorithms/algos/VerticalLinearRegression
   ./algorithms/algos/VerticalPoissonRegression
   ./algorithms/algos/VerticalXgboost
   ./algorithms/algos/VerticalXgboostDistributed
   ./algorithms/algos/VerticalBinningWoeIV
   ./algorithms/algos/VerticalFeatureSelection
   ./algorithms/algos/VerticalKMeans
   ./algorithms/algos/VerticalPearson
   ./algorithms/algos/VerticalSampler
   ./algorithms/algos/LocalNormalization
   ./algorithms/algos/LocalStandardScaler
   ./algorithms/algos/LocalDataSplit
   ./algorithms/algos/LocalFeaturePreprocess
   ./algorithms/algos/LocalDataStatistic

.. toctree::
   :maxdepth: 2
   :caption: 开发指南

   API <development/api.rst>
   算子开发 <development/algos_dev.rst>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
