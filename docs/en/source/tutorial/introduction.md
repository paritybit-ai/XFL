# Introduction

## Federated Learning
Federated learning is a distributed machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. 

### Features
- Data Isolation: The data of each participant is kept locally, the privacy will not be leaked. The data is available but invisible.
- Equal Benefits: All participants jointly establish a common model, with equal status and common benefits.
- Lossless: The modeling effect is basically close to or equivalent to the effect of modeling in one place.

### Category
- Horizontal Federated Learning: There is abundant features overlap among participant datasets, but lack of samples overlap.
- Vertical Federated Learning: There is abundant samples overlap among participant datasets, but lack of features overlap.
- Federated Transfer Learning: Both the samples overlap and the features overlap are lacking.

In our framework, algorithms are divided into three categories according to the characteristics of the data source:

1. `Local`(or `standalone`): The data is completely local, and the algorithm runs independently.
2. `Horizontal`：Horizontal Federated Learning. The data samples are distributed among different parties, and there is a shared data feature set.
3. `Vertical`：Vertical Federated Learning. Data features are distributed among different parties, and data samples are shared or overlapped.

For example: the training process of Horizontal Federated Learning is:

0. The server distributes the public key to the participants.
1. All participants use local data to train the model and calculate their own gradients, which are encrypted and transmitted to the server.
2. The server aggregates the gradients of each participant, updates the model parameters, and sends them to each participant.
3. Each participant updates its own model.
4. Perform 1-3 rounds of iterations.

The schematic diagram is as follows:
![](../images/Sect1.4HorizontalFL.png)

## Participant Role
### Scheduler
The overall task scheduler for the federated learning training process, assists in cluster networking, controls the Trainer, and is responsible for the distribution and management of Federated Learning tasks.

### Trainer
The execution node of the joint model training. Trainer is scheduled by Scheduler. All participants of Federated Learning will have one or more Trainers that communicate with the federated parameter server.
Usually include:
- trainer: Participant training node, usually the Federated Learning initiator. The dataset does not contain labels.
- label trainer: Participant training node, usually the Federated Learning participant. The dataset contains labels.

May include:
- assist trainer: A node used to assist training, such as horizontal gradient aggregation.


## Architecture



