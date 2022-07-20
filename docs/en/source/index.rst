.. XFL documentation master file, created by
   sphinx-quickstart on Thu Jul  7 17:24:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XFL's documentation!
===============================

XFL is a high-performance, high-flexibility, high-applicability, lightweight, open and easy-to-use Federated Learning framework. It supports various federation models such as horizontal federation and vertical federation. XFL uses homomorphic encryption, differential privacy, secure multi-party computing and other encryption computing technologies to protect users' original data from being leaked, and uses secure communication protocols to protect communication security, enabling users to conduct joint modeling on the basis of legal compliance to achieve data value.

Features:
  - High-performance algorithm library

    - Comprehensive algorithms: Support a variety of mainstream horizontal/vertical federation algorithms.
    - Excellent performance: The performance greatly exceeds the average level of the performance evaluation of CAICT(China Academy of Information and Communications Technology).
    - Network optimization: The training task can be completed in the case of weak network, high latency, massive packet loss, and long-term network disconnection.

  - Flexible application deployment

    - Flexible computing nodes: Support two-party/multi-party computing nodes deployment.
    - Flexible allocation of computing power: With or without label, any party can be the initiator, and assist computing nodes can be deployed on any party.
    - Flexible installation and deployment: Support CPU/GPU/hybrid deployment.

  - Lightweight and open

    - Lightweight: Low requirements on server performance, and some algorithms can be run in environments with poor performance.
    - Open: Support mainstream machine learning frameworks such as Pytorch and Tensorflow, and support user-defined horizontal models.


Function support
----------

==================================    ==================
             Function                   Implementation
==================================    ==================
      Horizontal Federation                   ✅
       Vertical Federation                    ✅
             XGBoost                          ✅
     Deep Learning Framework          Pytorch/Tensorflow
  Partial Homomorphic Encryption              ✅
   Fully Homomorphic Encryption               ✅
           One Time Pad                       ✅
  Multi-party Secure Computation              ✅
       Differential Privacy                   ✅
   PSI(Private Set Intersection)              ✅
PIR(Private Information Retrieval)            ✅
           GPU Support                        ✅
        Cluster Deployment                    ✅
         Online Inference                     ✅
    Federated Node Management                 ✅
    Federated Data Management                 ✅
==================================    ==================


.. toctree::
   :maxdepth: 2
   :caption: TUTORIAL

   Introduction <tutorial/introduction.md>
   Usage <tutorial/usage.md>

.. toctree::
   :maxdepth: 2
   :caption: ALGORITHMS

   Algorithms List <algorithms/algorithms_list.rst>
   Cryptographic Algorithm <algorithms/cryptographic_algorithm.rst>
   ./algorithms/differential_privacy
   ./algorithms/algos/HorizontalLinearRegression
   ./algorithms/algos/HorizontalLogisticRegression
   ./algorithms/algos/HorizontalResNet
   ./algorithms/algos/VerticalLogisticRegression
   ./algorithms/algos/VerticalXgboost
   ./algorithms/algos/VerticalBinningWoeIV
   ./algorithms/algos/VerticalFeatureSelection
   ./algorithms/algos/VerticalKMeans
   ./algorithms/algos/VerticalPearson
   ./algorithms/algos/LocalNormalization
   ./algorithms/algos/LocalStandardScaler

.. toctree::
   :maxdepth: 2
   :caption: DEVELOPMENT

   API <development/api.rst>
   Development Guide <development/algos_dev.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
