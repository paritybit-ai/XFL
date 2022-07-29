# Introduction

## Federated Learning
Federated learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples without exchanging them. 


### Features
- Data Isolation: The data of each participant is kept locally to prevent the leakage of privacy. The data is available while invisible.
- Equal Benefits: All participants jointly establish a shared model, with equal status and common benefits.
- Lossless: The modeling effect is basically close to or equivalent to the effect of modeling in one place.

### Category
- Horizontal Federated Learning: Participant data sets share the same feature space but different in data samples.
- Vertical Federated Learning: Participant data sets share the same sample ID space but different in feature space.
- Federated Transfer Learning: Participant data sets are different in both sample ID space and feature space.

In our framework, algorithms are classified into three types according to the data source:

1. `Local`(or `Standalone`): The data is completely local, and the algorithm runs independently.
2. `Horizontal`：Horizontal Federated Learning. Data samples are distributed among different parties while sharing the same data features.
3. `Vertical`：Vertical Federated Learning. Data features are distributed among different parties while data samples are shared or overlapped.

For example, the training process of Horizontal Federated Learning is:

0. The server distributes the public key to the participants.
1. All participants use their local data to train the model and calculate their own gradients, which are encrypted and transmitted to the server.
2. The server aggregates the gradients of each participant, updates the model parameters, then sends them to each participant.
3. Each participant updates its own model.
4. Repeat steps 1-3 till the iterations end.

The schematic diagram is as follows:
![](../images/Sect1.4HorizontalFL.png)

## Participant Role
### Scheduler
The overall task scheduler for the federated learning training process, assisting in cluster networking, controling the Trainer, and responsible for the distribution and management of Federated Learning tasks.

### Trainer
The execution node of the joint model training. Trainer is scheduled by Scheduler. All participants of Federated Learning will have one or more Trainers that communicate with the federated parameter server.
Usually include:
- trainer: Participant training node, usually the Federated Learning initiator. The dataset does not contain labels.
- label trainer: Participant training node, usually the Federated Learning participant. The dataset contains labels.

May include:
- assist trainer: A node used to assist training, such as horizontal gradient aggregation.


## Architecture diagram



