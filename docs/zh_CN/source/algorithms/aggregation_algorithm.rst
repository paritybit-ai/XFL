=======================
横向聚合算法
=======================

简介
============
在每一个全局模型迭代周期中，典型的横向联邦学习主要涉及到以下三个步骤：

1. 服务端将全局模型广播给每一个参与方；
2. 参与方使用本地数据集对模型进行训练，完成训练后将模型发送给服务端；
3. 服务端接收到所有参与方的模型，将它们聚合成为新的全局模型。

在算法层面上，以上三个步骤中的第三步是联邦学习中最为关键的部分，因为它决定了全局模型的训练效率和稳定性。
有多种聚合算法可以将参与方的模型聚合成为新的全局模型，其中最为常用的是 fedavg，它的优点是简单易用，且通信量较小。
但是，fedavg 也存在以下问题：

1. 在数据为非独立同分布（即异质性数据）时，无法保证全局模型的收敛；
2. 当存在拜占庭参与方（即故意上传错误数据的参与方）时，训练会变得非常不稳定；
3. 设备差异、通信效率、对参与方掉线的鲁棒性等一系列其他问题。

许多已发表的横向聚合算法旨在取代 fedavg，以提高模型训练的性能。各种主流聚合算法已在 XFL 中实现，灵活且易于使用。
数据科学家可以选择适当的聚合算法来完成特定的联邦学习任务。

聚合类型
================
不同的算法可能适用于不同的数据集和实验条件。通常来说， **fedavg** 都可以有良好的表现，但是当训练遇到瓶颈时，
可以尝试其他聚合算法。

fedavg
------
**fedavg** [fedavg]_ 是其他聚合算法的基础，可以被描述为以下过程:

| **输入**: 参与方数量 :math:`P`，全局迭代次数 :math:`T`，本地迭代次数 :math:`E`，损失函数 **loss function**，优化器 **optimizer** 
| **输出**: 单个全局模型 :math:`M^T`
| **服务端执行**:
|   **for** :math:`t=0,1,...,T-1` **do**
|     广播 :math:`M^t` 给每一个参与方
|     **for** :math:`i=0,1,...,P-1` **in parallel do**
|       执行本地训练，上传本地模型 :math:`m^i` 和聚合权重 :math:`w^i`
|     :math:`M^{t+1} \leftarrow \frac{\sum_iw^im^i}{\sum_iw^i}`
|   **返回** :math:`M^T`
| **参与方执行**:
|   :math:`m^i \leftarrow M^t`
|   **for** epoch :math:`k=0,1,...,E-1` **do**
|     **for** each batch **do**
|       根据损失函数 **loss function** 和优化器 **optimizer** 更新参数
|   计算聚合权重 :math:`w^i`
|   **返回** :math:`m^i` 和 :math:`w^i`

在 **fedavg** 原本的设计中，使用的优化器 **optimizer** 是随机梯度下降(SGD)，
使用的聚合权重 :math:`w^i` 等于本地数据集的分批数量。而XFL提供了更强大和灵活的配置方式，
XFL 允许配置任意的优化器 **optimizer** ，例如 Adam，并且用户可以自由地改变聚合权重的定义。

fedprox
-------
**fedprox** [fedprox]_ 是基于 **fedavg** 实现的聚合算法，可能在数据为非独立同分布的情况下提升模型的表现。
对于任意的损失函数 **loss function** :math:`L`， **fedprox** 会自动地给它添加一个正则项 
:math:`\frac{\mu}{2}||m^i-M^t||^2`。因此，实际的损失函数 **loss function** 
变为 :math:`L + \frac{\mu}{2}||m^i-M^t||^2`。 :math:`\mu` 是需要人为给定的超参数，例如：

::

    "type": "fedprox", 
    "mu": 1,

scaffold
--------
**scaffold** [scaffold]_ 是基于 **fedavg** 实现的聚合算法，可能在数据为非独立同分布的情况下提升模型的表现。
**scaffold** 独立实现了优化器 **optimizer**，因此在使用 **scaffold** 时手动指定的其他优化器 **optimizer** 
会被忽略。


.. [fedavg] McMahan B., Moore E., Ramage D. et al, Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of AISTATS, pp. 1273-1282, 2017.
.. [fedprox] Li T., Sahu A. K., Zaheer M. et al. Federated optimization in heterogeneous networks. In MLSys, 2020.
.. [scaffold] Karimireddy S. P., Kale S., Mohri M. et al. SCAFFOLD: Stochastic controlled averaging for on-device federated learning. In Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020.