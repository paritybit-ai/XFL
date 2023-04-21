===========
算子开发
===========

创建算子文件
============

在XFL中创建算子前，首先需要创建算子所在目录和文件。XFL中内置算子均存放在 `python/alogorithm/framework` 文件夹下。
在创建新算子时，需要按照"联邦类型" [#type]_ /"算法名"来创建算子文件夹，然后根据算子所需的“联邦角色” [#role]_ 创建.py文件。
以创建纵向Xgboost算子为例，算子联邦类型为vertical，算法名称为xgboost, 该算子包含两种角色：label_trainer和trainer，因此创建的文件目录结构如下：

::

    | — vertical
    | | — xgboost
    | | | — label_trainer.py
    | | | — trainer.py


创建算子类
============

创建好算子文件后，接下来需要为每个联邦角色创建算子类。XFL支持算子的自动发现，需要要求算子类的命名遵守如下的规范。
通过如下步骤得到算子名称：

1. 将联邦类型，算法名，联邦角色用下划线'_'连接；
#. 将首字母和下划线后的字母改为大写;
#. 删除下划线得到类名。

以纵向Xgboost算子为例，需要在label_trainer.py中创建类VerticalXgboostLabelTrainer, 在trainer.py中创建类VerticalXgboostTrainer. 
所有算子接受相同的参数train_info, 该参数为一个词典类型. 每个类中必须包含fit方法，
通过该方法来进行算子的训练, fit方法不接收其他参数。例如：

.. code-block:: python

    class VerticalXgboostLabelTrainer():
        def __init__(self, train_conf):
            pass

        def fit(self):
            pass


.. code-block:: python

    class VerticalXgboostTrainer():
        def __init__(self, train_conf):
            pass

        def fit(self):
            pass


开发算子
=============

算子入参
-------------

算子接收train_conf作为输入参数，train_conf为dict类型，内容与用户调用算子时的参数一致。 train_conf中主要信息如下：

.. code-block:: json

    {
        "identity": "label_trainer",
        "model_info": {
            "name": "vertical_xgboost",
            "config": {

            }
        },
        "input": {

        },
        "output": {

        },
        "train_info": {

        }
    }

其中，identity规定了算子调用方的身份，label_trainer, trainer, 或assist_trainer, model_info.name为算子名，
由算子类型horizontal, vertical或local和算法名称组成，以下划线连接。input中为算子输入数据信息，output为算子输出数据信息，train_info为算子训练所需信息。

算子结构
-------------

算子包含__init__方法和fit方法。XFL通过__init__初始化算子，通过fit方法调用算子训练。我们建议将初始化数据集，模型，loss，metric, optimizer，
通信信道等一次性操作的代码放在__init__中，将模型训练代码放在fit方法中。

开发工具
=============

通信模块
-------------

XFL基于grpc+redis封装了一个简洁的通信模块，该模块提供点对点、广播通信两种模式。开发者可以通过该模块创建信道，发送和接收数据。

1. 点对点通信

- 创建channel

.. code-block:: 

    class DualChannel(name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True)
    
        Args:
            name (str): 通道名称.
            ids (list): 通信双方id列表.
            job_id (Union[str, int], optional): 联邦任务id，默认自动获取. Defaults to "".
            auto_offset (bool, optional): 是否自动累加通信次数, 当为False时，应在传输时手动输入tag并确保不同轮次通信的tag不重复. 
                Defaults to True.
   

- 发送数据

.. code-block:: 

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int

        Args:
            value (Any): 发送的数据, 任意类型。
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.

        Returns:
            int: 0表示发送成功，否则发送失败。

- 接收数据

.. code-block:: 

    recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any
    
        Args:
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据反序列化，应与发送方参数设置相同. Defaults to True.
            wait (bool, optional): 是否等待接收完成，若为False，则立即返回. Defaults to True.

        Returns:
            Any: 若wait为True，返回相同轮次或tag相同的发送端数据；若wait为False，若已完成接收完整数据，则返回已接收数据，
                否则返回None.
        
- 交换数据

.. code-block:: 

    swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any

        Args:
            value (Any): 本方发送的数据，任意类型。
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将发送数据序列化和接收数据反序列化. Defaults to True.

        Returns:
            Any: 对方发送的数据


:Example:

假设联邦中只有一个label trainer和一个trainer.

- trainer

.. code-block:: 

    from common.communication.gRPC.python.channel import DualChannel
    from service.fed_config import FedConfig

    demo_chann = DualChannel(name="demo_dual_chann", ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
    demo_chann.send(1)
    b = demo_chann.swap(2) 
    # b = 3


- label trainer

.. code-block:: 

    from common.communication.gRPC.python.channel import DualChannel
    from service.fed_config import FedConfig

    demo_chann = DualChannel(name="demo_dual_chann", ids=[FedConfig.node_id] + FedConfig.get_trainer())
    a = demo_chann.recv() 
    # a = 1
    b = demo_chann.swap(3) 
    # b = 2


2. 广播通信

- 创建channel

.. code-block:: 

    class BroadcastChannel(name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True)
    
        Args:
            name (str): 通道名称.
            ids (List[str], optional): 所有通信参与方的id列表，默认自动获取所有当前联邦参与方id. Defaults to [].
            root_id (str, optional): 广播信道中的中心节点，默认自动获取label trainer的id作为root_id. Defaults to ''.
            job_id (Union[str, int], optional): 联邦任务id，默认自动获取. Defaults to "".
            auto_offset (bool, optional): 是否自动累加通信次数，当为False时，应在传输时手动输入tag并确保不同轮次通信的tag不重复.
                Defaults to True.

-  root节点广播数据

.. code-block:: 

    broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int

        Args:
            value (Any): 广播数据，任意类型.
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.

        Returns:
            int: 0表示发送成功，否则发送失败。


- root节点分发数据（其他节点收到数据不相同）

.. code-block:: 

    scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int

        Args:
            values (List[Any]): 分发数据。list长度等于其他通信节点的数量，第i个位置的数据发送到第i个通信节点。通信节点的顺序
                与初始化时ids中的节点顺序一致（不包括root节点）。
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.

        Returns:
            int: 0表示发送成功，否则发送失败。

- root节点接收数据

.. code-block:: 

    collect(tag: str = '@', use_pickle: bool = True) -> List[Any]

        Args:
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.

        Returns:
            List[Any]: 收到的数据。list长度等于其他通信节点的数量，第i个位置的数据为第i个通信节点发送的数据。通信节点的顺序
                与初始化时ids中的节点顺序一致（不包括root节点）。

- 非root节点发送数据到root

.. code-block:: 

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int

        Args:
            value (Any): 发送的数据，任意类型.
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.
        Returns:
            int: 0表示发送成功，否则发送失败。

- 非root节点从root接收数据

.. code-block:: 

    recv(tag: str = '@', use_pickle: bool = True) -> Any

        Args:
            tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
            use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. 
                Defaults to True.

        Returns:
            Any: 收到的数据.


:Example:

以root节点为assist_trainer，其他节点为trainer节点, 顺序为node-1, node-2为例。

- assist_trainer

.. code-block:: 

    from common.communication.gRPC.python.channel import BroadcastChannel
    from service.fed_config import FedConfig

    demo_chann = BroadcastChannel(name='demo_broadcast_chann',
                                ids=FedConfig.get_trainer() + [FedConfig.get_assist_trainer()],
                                root_id=FedConfig.get_assist_trainer())

    demo_chann.broadcast(1)
    demo_chann.scatter([2, 3])
    a = demo_chann.collect()
    # a = [4, 5]

- trainer: node-1

.. code-block:: 

    from common.communication.gRPC.python.channel import BroadcastChannel
    from service.fed_config import FedConfig

    demo_chann = BroadcastChannel(name='demo_broadcast_chann',
                                ids=FedConfig.get_trainer() + [FedConfig.get_assist_trainer()],
                                root_id=FedConfig.get_assist_trainer())
    a = demo_chann.recv()
    # a = 1
    a = demo_chann.recv()
    # a = 2
    demo_chann.send(4)

- trainer: node-2

.. code-block:: 

    from common.communication.gRPC.python.channel import BroadcastChannel
    from service.fed_config import FedConfig

    demo_chann = BroadcastChannel(name='demo_broadcast_chann',
                                ids=FedConfig.get_trainer() + [FedConfig.get_assist_trainer()],
                                root_id=FedConfig.get_assist_trainer())
    a = demo_chann.recv()
    # a = 1
    a = demo_chann.recv()
    # a = 3
    demo_chann.send(5)

聚合模块
-------------

聚合模块的参与方分为两种：root和leaf，其中root为中心节点，可进行参数的广播和聚合；leaf为非中心节点，可进行参数的上传和下载。
聚合模块支持明文聚合和密文聚合，其中密文聚合目前支持一次一密 [#FedAvg]_ 的加密方式。

1. 中心节点

XFL支持两种中心节点的初始化，分别是AggregationPlainRoot和AggregationOTPRoot，其中AggregationOTPRoot是支持一次一密的加密聚合。

- 创建实例

.. code-block:: 

    get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]

        Args:
            sec_conf (dict): 安全参数。包含关键字method, 值为plain或者otp. 若为otp，则还应包含otp的配置参数，具体的参数见下方示例。
            root_id (str, optional): 中心节点id. 默认取assist_trainer的id. Defaults to ''.
            leaf_ids (list[str], optional): 非中心节点id列表. 默认取label_trainer和trainer的并集. Defaults to [].

        Returns:
            Union[AggregationPlainRoot, AggregationOTPRoot]: 根据sec_conf的配置返回AggregationPlainRoot或AggregationOTPRoot的实例。

sec_conf的示例如下：

**明文的配置：**

.. code-block:: json

    {
        "method": "plain"
    }
    
**一次一密的配置：**

.. code-block:: json

    {
        "method": "otp",
        "key_bitlength": 64,
        "data_type": "torch.Tensor",
        "key_exchange": {
            "key_bitlength": 3072,
            "optimized": true
        },
        "csprng": {
            "name": "hmac_drbg",
            "method": "sha512"
        }
    }

中心节点包含以下方法：

- 设置root节点待分发的原始全局数据

.. code-block:: 

    set_initial_params(params: OrderedDict) -> None

        Args:
            params (OrderedDict): 原始全局数据.

- 接收非中心节点数据并计算聚合数据，聚合公式为: :math:`\sum_{i} parameters_i \cdot parameters\_weight_i`

.. code-block:: 

    aggregate() -> OrderedDict

        Returns:
            OrderedDict: 聚合后的数据.

- 广播数据到所有非中心节点

.. code-block:: 

    broadcast(params: OrderedDict) -> int

        Args:
            params (OrderedDict): 待分发的全局数据.

        Returns:
            int: 0表示广播成功，否则失败.

2. 非中心节点

与中心节点对应，非中心节点也包含两种：AggregationPlainLeaf和AggregationOTPLeaf. 通过如下函数初始化：

- 创建实例

.. code-block:: 

    get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]

        Args:
            sec_conf (dict): 安全参数。与get_aggregation_leaf_inst中sec_conf参数相同。
            root_id (str, optional): 中心节点id. 默认取assist_trainer的id. Defaults to ''.
            leaf_ids (list[str], optional): 非中心节点id列表. 默认取除label_trainer和trainer的并集. Defaults to [].

        Returns:
            Union[AggregationPlainLeaf, AggregationOTPLeaf]: 根据sec_conf的配置返回AggregationPlainLeaf或AggregationOTPLeaf的实例。

非中心节点包含以下方法：

- 上传数据和数据的权重到中心节点

.. code-block:: 

    upload(parameters: OrderedDict, parameters_weight: float) -> int

        Args:
            parameters (OrderedDict): 要上传的数据.
            parameters_weight (float): 上传数据的权重.

        Returns:
            int: 0表示上传成功，否则失败.

- 从中心节点下载数据

.. code-block:: 

    download() -> OrderedDict

        Returns:
            OrderedDict: 下载数据.




横向联邦算子开发
================

与纵向联邦不同，横向联邦通信模式一般比较固定，XFL提供了预置的模版类，开发者可以使用模版快速开发横向模型。
目前XFL提供了基于FedAvg的横向联邦模版类，
见 `FedAvgTemplateAssistTrainer <../../../../python/algorithm/core/horizontal/template/torch/fedavg/assist_trainer.py>`_ , 
`FedAvgTemplateLabelTrainer <../../../../python/algorithm/core/horizontal/template/torch/fedavg/label_trainer.py>`_ , 使用该模版的开发实例可参考 `HorizontalLogisticRegressionAssistTrainer <../../../../python/algorithm/framework/horizontal/logistic_regression/assist_trainer.py>`_ , `HorizontalLogisticRegressionLabelTrainer <../../../../python/algorithm/framework/horizontal/logistic_regression/label_trainer.py>`_ .


:说明:

.. [#type] XFL目前支持三种算子类型：horizontal, vertical和local.
.. [#role] XFL支持三种联邦角色：assist_trainer, label_trainer和trainer.
.. [#FedAvg] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]//proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 1175-1191.
