[![License](https://img.shields.io/github/license/paritybit-ai/XFL)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/xfl/badge/?version=latest)](https://xfl.readthedocs.io/en/latest/?badge=latest)


XFL is a high-performance, high-flexibility, high-applicability, lightweight, open and easy-to-use Federated Learning framework.
It supports a variety of federation models in both horizontal and vertical federation scenarios. 
To enable users to jointly train model legally and compliantly to unearth the value of their data, XFL adopts homomorphic encryption,
differential privacy, secure multi-party computation and other security technologies to protect users' local data from leakage,
and applies secure communication protocols to ensure communication security.

# Highlights

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


# Quick Start Demo

Running in standalone mode

```shell
# create and activate the virtual environment
conda create -n xfl python=3.9.7
conda activate xfl

# install redis and other dependencies
# Ubuntu
apt install redis-server
# CentOS
yum install epel-release
yum install redis
# MacOS
brew install redis
brew install coreutils

# install python dependencies
# update pip
pip install -U pip
# install dependencies
pip install -r requirements.txt

# set permission
sudo chmod 755 /opt

# enter the project directory
cd ./demo/vertical/logistic_regression/2party

# start running the demo
sh run.sh
```

- [Quick Start](./docs/en/source/tutorial/usage.md)
# Document

- [Document](https://xfl.readthedocs.io/en/latest)
## Tutorial
- [Introduction](./docs/en/source/tutorial/introduction.md)

## Algorithms
- [List of Availble Algorithms](./docs/en/source/algorithms/algorithms_list.rst)
- [Cryptographic Algorithms](./docs/en/source/algorithms/cryptographic_algorithm.rst)
- [Differential Privacy](./docs/en/source/algorithms/differential_privacy.rst)

## Development
- [API](./docs/en/source/development/api.rst)
- [Developer Guide](./docs/en/source/development/algos_dev.rst)

# License
[Apache License 2.0](./LICENSE)
