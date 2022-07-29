=======================
Differential Privacy
=======================

Introduction
--------------

In our framework, differential privacy aimes to solve this problem: how can we train a model that has good performance 
in the population while protecting privacy at the individual level.

The basic idea of differential privacy is: add noise in the model training process, such that the model trained on two datasets 
that differ by one record are almost indistinguishable.

.. image:: ../images/dp_orig_algos.png
    :width: 45%
.. image:: ../images/dp_preserving_algos.png
    :width: 45%

As shown in the left figure above, the algorithm maps the dataset into the parameter space. 
Because of some randomness (such as that in the initialization of neural network), the learned parameter has a probility distribution. 
Normally, if two datasets are different, the probility distribution of their learned models may be distinguishable (left figure). 
This means that, we can spot the difference between two datasets through the difference of the learned model parameters. 
Consider an extreme case where dataset D1 is identical to dataset D2 except one record in plus. 
Then by inspecting the difference in the output models, one can determine if this personal record is in the dataset. 
After introducing noise by differential privacy, the distribution of model parameters become indistinguishable. 
Then it is hard to tell if this specific record is in the dataset.

Mathematical Definition
-----------------------

Now we introduce some mathematical definition to quantify the previous hand-waving argument. 
A common definition is the :math:`(\epsilon, \delta)` -differential privacy [Dwork2014]_  


  A randomized algorithm :math:`\mathcal{M}` with domain :math:`\mathbb{N}^{|\mathcal{x}|}` is 
  :math:`(\epsilon, \delta)` -differentially private if for all :math:`\mathcal{S} \in Range(\mathcal{M})` 
  and for all :math:`D_1, D_2 \in \mathbb{N}^{|\mathcal{X}|}` such that :math:`||D_1 - D_2||_1 \leq 1` :

  .. math::
    Pr[\mathcal{M}(D_1) \in \mathcal{S}] \leq \exp(\epsilon) Pr[\mathcal{M}(D_2) \in \mathcal{S}] + \delta

  -- Dwork and Roth

In the above definition, the domain :math:`\mathbb{N}^{|\mathcal{X}|}` is all possible datasets, 
and :math:`D_1` and :math:`D_2` are two datasets. :math:`Range(\mathcal{M})` is the algorithm output, 
such as trained neural networks. The constraint :math:`||D_1-D_2||_1 \leq 1` means that they differ by only one record.

Differential Privacy in Federated Learning
-------------------------------------------

In our framework, we implement the algorithm "Noise before Model aggregation FL" (NbAFL [Wei2020]_ ). 
In short, we add approriate noise in both uplink and downlink channel when performing FedAvg to protect data privacy.


.. [Dwork2014] C. Dwork and A. Roth, The Algorithmic Foundations of Differential Privacy, Foundations and Trends in Theoretical Computer Sicence, Vol 9, (2014) 211-407
.. [Wei2020] K. Wei et al., Federated Learning with Differential Privacy: Algorithms and Performance Analysis, IEEE Transactions on Information Forensics and Security, 1556-2021 (2020) 