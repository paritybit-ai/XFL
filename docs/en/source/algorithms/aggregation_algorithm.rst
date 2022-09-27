=======================
Aggregation Algorithms
=======================

Introduction
============
In each global epoch, the typical horizontal federated learning paradigm involves three stages:

1. the server broadcasts the global model to every client.
2. clients conduct model training with their local datasets.
3. the server gathers the local models and aggregates them to obtain a global model.

One of the standard aggregation methods is fedavg, which is widely used due to its simplicity and low communication. 
Extensive experiments have proved the availability and stability of fedavg, but many problems also emerged:

1. convergence is not guaranteed for non-IID (heterogeneous) data.
2. training is very unstable when there are Byzantine clients (i.e. clients who intentionally upload wrong parameters).
3. heterogeneous devices, communication efficiency, tolerance to dropped clients and other aspects to be improved.

Many published works are designed to replace fedavg to improve the performance of model training. 
Various mainstream aggregation algorithms have been implemented in XFL, flexibly and easy to use.
Data scientists are able to choose appropriate aggregation algorithms to deal with specific federated learning tasks.

Aggregation Type
================
Different algorithms may be suitable for different datasets and experimental conditions. Usually, **fedavg** performs well,
but one can try other algorithms when the training meets a bottleneck.

fedavg
------
**fedavg** [fedavg]_ is the base of the other algorithms, which can be described as follows:

| **input**: number of parties :math:`P`, global epoch :math:`T`, local epoch :math:`E`, **loss function**, **optimizer** 
| **output**: Single global model :math:`M^T`
| **server executes**:
|   **for** :math:`t=0,1,...,T-1` **do**
|     broadcast :math:`M^t` to all clients
|     **for** :math:`i=0,1,...,P-1` **in parallel do**
|       conduct local training, upload local model :math:`m^i` and aggregation weight :math:`w^i`
|     :math:`M^{t+1} \leftarrow \frac{\sum_iw^im^i}{\sum_iw^i}`
|   **return** :math:`M^T`
| **party executes**:
|   :math:`m^i \leftarrow M^t`
|   **for** epoch :math:`k=0,1,...,E-1` **do**
|     **for** each batch **do**
|       update parameters according to the **loss function** and **optimizer**
|   calculate the aggregation weight :math:`w^i`
|   **return** :math:`m^i` and :math:`w^i` to the server

In the prototype of **fedavg**, the **optimizer** is stochastic gradient descent(SGD), and the aggregation weight :math:`w^i` is equal to 
the number of local batches. However, XFL admits arbitrary **optimizer** such as Adam, and users can freely change the definition of aggregation weights.

fedprox
-------
**fedprox** [fedprox]_ is implemented based on **fedavg**, which may improve the training performance when the data is non-IID. 
For arbitrary **loss function** :math:`L`, **fedprox** automatically adds a regularizer :math:`\frac{\mu}{2}||m^i-M^t||^2`. 
Therefore, the actual **loss function** becomes :math:`L + \frac{\mu}{2}||m^i-M^t||^2`. :math:`\mu` is a superparameter needs to be given, just like

::

    "type": "fedprox", 
    "mu": 1,


.. [fedavg] McMahan B., Moore E., Ramage D. et al, Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of AISTATS, pp. 1273-1282, 2017.
.. [fedprox] Li T., Sahu A. K., Zaheer M. et al. Federated optimization in heterogeneous networks. In MLSys, 2020.