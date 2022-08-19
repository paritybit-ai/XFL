.. XFL documentation master file, created by
   sphinx-quickstart on Thu Jul  7 17:24:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XFL's documentation!
===============================

XFL is a high-performance, high-flexibility, high-applicability, lightweight, open and easy-to-use Federated Learning framework.
It supports a variety of federation models in both horizontal and vertical federation scenarios. 
To enable users to jointly train model legally and compliantly to unearth the value of their data, XFL adopts homomorphic encryption,
differential privacy, secure multi-party computation and other security technologies to protect users' local data from leakage,
and applies secure communication protocols to ensure communication security.

Highlights:

  - High-performance algorithm library

    - Comprehensive algorithms: support a variety of mainstream horizontal/vertical federation algorithms.
    - Excellent performance: significantly exceeds the average performace of federated learning products. 
    - Network optimization: adapt to high latency, frequent packet loss, and unstable network environments.

  - Flexible deployment

    - parties: support two-party/multi-party federated learning.
    - schedulering: any participant can act as a task scheduler.
    - hardware: support CPU/GPU/hybrid deployment.

  - Lightweight, open and easy to use:

    - Lightweight: low requirements on host performance.
    - Open: support mainstream machine learning frameworks such as Pytorch and Tensorflow, and user can conveniently design their own horizontal federation models.
    - Easy to use: able to run in both docker environment and Conda environment.

Function support
--------------------

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
  Secure Multi-party Computation              ✅
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
   Quickstart Guide <tutorial/usage.md>

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
   ./algorithms/algos/VerticalSampler
   ./algorithms/algos/LocalNormalization
   ./algorithms/algos/LocalStandardScaler
   ./algorithms/algos/LocalDataSplit
   ./algorithms/algos/LocalFeaturePreprocess

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
