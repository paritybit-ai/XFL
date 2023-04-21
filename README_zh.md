### [English](./README.md) 

[![License](https://img.shields.io/github/license/paritybit-ai/XFL)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/xfl/badge/?version=latest)](https://xfl.readthedocs.io/en/latest/?badge=latest)
[![coverage report](https://git.basebit.me/bag1/federated-learning/badges/master/coverage.svg)](https://git.basebit.me/bag1/federated-learning/-/commits/master)

XFL是一个高性能、高灵活度、高适用性、轻量开放的联邦学习框架，支持横向联邦、纵向联邦等多种联邦模型，运用同态加密、差分隐私、多方安全计算等多种加密计算技术保护用户的原始数据不被泄露，使用安全通信协议保护通信安全，使用户在合法合规的基础上进行联合建模，实现数据价值。

# 项目特点
  - 高性能的算法库

    - 算法全面：支持多种主流横向/纵向联邦算法
    - 性能优异：性能大幅领先于信通院性能评测平均水平
    - 网络优化：在弱网络、高延迟、大量丢包、较长时间断网的情况下依然可以完成训练任务

  - 应用部署灵活

    - 计算节点灵活： 支持两方/多方计算节点部署
    - 算力调配灵活： 无论有无label，支持任何一方为发起方，支持辅助计算节点部署在任一方
    - 安装部署灵活： 支持CPU/GPU/混合部署

  - 轻量开放

    - 轻量： 对服务器性能要求低，部分算法可在性能较差的环境下运行
    - 开放：支持 Pytorch / Tensorflow 等主流机器学习框架，支持用户自定义横向模型

# [快速开始](./docs/zh_CN/source/tutorial/usage.md)

# Document
- [Document](https://xfl.readthedocs.io/en/latest)

## Tutorial
- [简介](./docs/en/source/tutorial/introduction.md)

## Algorithms
- [算法列表](./docs/en/source/algorithms/algorithms_list.rst)
- [密码算法](./docs/en/source/algorithms/cryptographic_algorithm.rst)
- [差分隐私](./docs/en/source/algorithms/differential_privacy.rst)

## Development
- [API](./docs/en/source/development/api.rst)
- [开发指南](./docs/en/source/development/algos_dev.rst)

# License
[Apache License 2.0](./LICENSE)
