# Demo执行方法
### 用例文件格式
以两方逻辑回归为例
```
horizontal/logistic_regression/2party
├── config
│   ├── fed_conf.json
│   ├── trainer_config_node-1.json
│   └── trainer_config_node-2.json
├── run.sh
└── stop.sh
```
* config: demo相关配置文件, 包括联邦学习配置、模型参数配置和训练参数配置.
* run.sh: demo启动脚本.
* stop.sh: demo终止脚本，用于终止运行和清理端口.

### 执行方法

1. 进入用例路径.
```
cd $PROJECT_HOME/demo/horizontal/logistic_regression/2party
```
2. 运行用例.
```
sh run.sh
```
3. 终止任务.
```
sh stop.sh
```

