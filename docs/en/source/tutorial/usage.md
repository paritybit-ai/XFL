# Basic Tutorial for XFL

We provide two methods for users to start running with XFL: via docker or conda.

### Prerequisites
* Python：3.9

* OS：MacOS，Linux（most distributions supported, CentOS 7 in this tutorial）

* CPU/Memory：8 cores and 16G memory is recommended as a minimum configuration

* Redis

### Clone the repository
```shell
git clone https://github.com/paritybit-ai/XFL
cd XFL
```

## Using docker

### Run redis-server
```shell
# Get the redis image
docker pull redis:7.0.3

# start a redis server, and make sure that port 6379 is open
docker run --name my_redis -p 6379:6379 -d redis:7.0.3
```

### Start with XFL
Run the XFL image as a container
```shell
# pull the image
docker pull basebit/xfl:1.2.0

# start with XFL
docker run -it --entrypoint /bin/bash \
--network container:my_redis \
-v $PWD/demo:/opt/app/xfl/demo \
-v /opt/dataset:/opt/dataset \
-v /opt/config:/opt/config \
-v /opt/log:/opt/log \
-v /opt/checkpoints:/opt/checkpoints \
basebit/xfl:1.4.0
```

### Quick start demo
Running inside docker contrainer 
```shell
cd demo/vertical/logistic_regression/2party
sh run.sh
```

## Using Conda

### Environment Preparation

It is recommended to use anaconda to create a virtual environment.

```shell
conda create -n xfl python=3.9.7

conda activate xfl
```
Install redis and other dependencies

```shell
# Ubuntu
apt install redis-server

# CentOS
yum install epel-release
yum install redis

# MacOS
brew install redis
brew install coreutils
```

install python dependencies
```shell
# update pip
pip install -U pip

# install dependencies
uv sync
```

### Quick start demo

Running in standalone mode
```shell
# set permission
sudo chmod 755 /opt

# enter the project directory
cd demo/vertical/logistic_regression/2party

# activate the virtual environment
conda activate xfl

# start redis-server (skip if it is already run)
redis-server & # for Ubuntu and MacOS

systemctl start redis # for CentOS

# start running the demo
sh run.sh
```

## Demonstration tutorial

We provide various examples in the `demo` directory of the project.

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

In each subdirectory, an executable script for demonstration is provided.
For example, the following commands run the vertical logistic regression (two parties)
```
cd demo/vertical/logistic_regression/2party

sh run.sh
```

* The read and write permissions to the `\opt` directory are required when running with the default configuration. If the permission cannot be obtained, one should modify the configuration under the corresponding subdirectory.  
* A `JOB_ID` will be automatically assigned to the task after it is running. According to this `JOB_ID`, the output and log files of the task can be obtained.
* After tasks are executed, the log `INFO: All Stage Successful.` will be printed, indicating that all tasks were executed successfully.

A successfully executed *vertical logistic regression (two parties)* produces the following files:
```
/opt
├── checkpoints         # model path
│   ├── ...
│   └── 4               # [JOB_ID]
│       ├── node-1      # output directory for node-1 (model file included)
│       │   ├── vertical_logistic_regression_guest.pt
│       │   └── ...     
│       └── node-2      # output directory for node-2 (model file included)
│           ├── vertical_logistic_regression_host.pt
│           └── ...     
└── log
    └── 4               # [JOB_ID]
        └── xfl.log     # log file
```
After the task is completed, one can clean up residual processes with following script.
```shell
# clean up residual processes
sh stop.sh
```
