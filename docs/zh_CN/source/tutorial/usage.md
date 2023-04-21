# XFL任务基本教程

我们提供了两种方式配置和运行XFL：通过docker或conda来搭建环境。

### 环境准备
* Python：3.9

* OS：MacOS，Linux（支持主流的发行版本，教程使用的是CentOS 7）

* CPU/Memory：推荐的最低配置为8核16G内存

* Redis

### 获取代码库
```shell
git clone https://github.com/paritybit-ai/XFL
cd XFL
```

## 使用Docker方式

### 启动Redis
```shell
# 获取redis镜像
docker pull redis:7.0.3

# 启动redis，注意需要保证6379的端口开放
docker run --name my_redis -p 6379:6379 -d redis:7.0.3
```

### 启动XFL
```shell
# 获取镜像
docker pull basebit/xfl:1.4.0

# 启动XFL
docker run -it --entrypoint /bin/bash \
--network container:my_redis \
-v $PWD/demo:/opt/app/xfl/demo \
-v /opt/dataset:/opt/dataset \
-v /opt/config:/opt/config \
-v /opt/log:/opt/log \
-v /opt/checkpoints:/opt/checkpoints \
basebit/xfl:1.2.0
```

### 快速开始
```shell
# 在镜像中运行demo脚本
cd demo/vertical/logistic_regression/2party
sh run.sh
```

## 使用Conda方式

### 安装依赖

推荐使用anaconda创建虚拟环境

```shell
conda create -n xfl python=3.9.7

conda activate xfl
```

通过pip安装依赖
```shell
# 升级到最新版本
pip install -U pip

# 通过requirements.txt安装依赖
pip install -r requirements.txt
```

### 快速开始

本地单机运行第一个demo算子
```shell
# 进入项目目录
cd ./demo/vertical/logistic_regression/2party

# 激活虚拟环境
conda activate xfl

# 启动redis服务（已经开启了可以跳过这一步）
redis-server &

# 运行demo
sh run.sh
```

## 测试用例教程

在项目的demo目录里，我们准备了丰富的测试用例：

```
demo
├── horizontal
│   ├── logistic_regression
│   └── resnet
├── local
│   ├── normalization
│   └── standard_scaler
├── vertical
│   ├── binning_woe_iv
│   ├── feature_selection
│   ├── kmeans
│   ├── logistic_regression
│   ├── pearson
│   └── xgboost
└── ...
```

每个子目录下包含配置文件和执行脚本。以两方的纵向逻辑回归为例，启动步骤如下：

```
cd ./demo/vertical/logistic_regression/2party

sh run.sh
```
* 以默认配置运行时需要拥有`/opt`目录的读写权限。如果无法获取该目录权限，则需要修改对应的配置文件。
* 开始运行任务后，会自动分配一个`JOB_ID`，根据这个`JOB_ID`可以查看任务的输出以及日志文件。
* 任务全部执行完毕后，打印日志`INFO: All Stage Successful.`表示所有任务执行成功。

一个成功执行的*两方纵向逻辑回归*任务会生成以下文件：
```
/opt
├── checkpoints         # 模型输出的存放路径
│   ├── ...
│   └── 4               # 本次执行的[JOB_ID]
│       ├── node-1      # 节点1的输出文件（包含模型文件）
│       │   ├── vertical_logistic_regression_guest.pt
│       │   └── ...     
│       └── node-2      # 节点2的输出文件（包含模型文件）
│           ├── vertical_logistic_regression_host.pt
│           └── ...     
└── log
    └── 4               # 本次执行的[JOB_ID]
        └── xfl.log     # 日志文件
```
任务完成后，可以用`demo`目录下的脚本清楚残留子进程。
```shell
# 任务结束后杀死残留子进程
sh stop.sh
```
