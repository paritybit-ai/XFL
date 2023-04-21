====
API
====

通信模块
===========

点对点通信
-----------

**class DualChannel(name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True)**

创建点对点通信模块
    
- name (str): 通道名称.
- ids (list): 通信双方id列表.
- job_id (Union[str, int], optional): 联邦任务id，默认自动获取. Defaults to "".
- auto_offset (bool, optional): 是否自动累加通信次数, 当为False时，应在传输时手动输入tag并确保不同轮次通信的tag不重复. Defaults to True.


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

发送数据

- value (Any): 发送的数据, 任意类型。
- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. Defaults to True.

- Returns: 
    - int: 0表示发送成功，否则发送失败。


**recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any**

接收数据

- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据反序列化，应与发送方参数设置相同. Defaults to True.
- wait (bool, optional): 是否等待接收完成，若为False，则立即返回. Defaults to True.

- Returns: 
    - Any: 若wait为True，返回相同轮次或tag相同的发送端数据；若wait为False，若已完成接收完整数据，则返回已接收数据，否则返回None.


**swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any**

交换数据

- value (Any): 本方发送的数据，任意类型。
- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将发送数据序列化和接收数据反序列化. Defaults to True.

- Returns:
    - Any: 对方发送的数据


广播通信
---------

**class BroadcastChannel(name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True)**
    
创建广播通信模块

- name (str): 通道名称.
- ids (List[str], optional): 所有通信参与方的id列表，默认自动获取所有当前联邦参与方id. Defaults to [].
- root_id (str, optional): 广播信道中的中心节点，默认自动获取label trainer的id作为root_id. Defaults to ''.
- job_id (Union[str, int], optional): 联邦任务id，默认自动获取. Defaults to "".
- auto_offset (bool, optional): 是否自动累加通信次数，当为False时，应在传输时手动输入tag并确保不同轮次通信的tag不重复. Defaults to True.


**broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

中心节点广播数据

- value (Any): 广播数据，任意类型.
- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. Defaults to True.

- Returns:
    - int: 0表示发送成功，否则发送失败。

**scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int**

中心节点分发数据

- values (List[Any]): 分发数据。list长度等于其他通信节点的数量，第i个位置的数据发送到第i个通信节点。通信节点的顺序与初始化时ids中的节点顺序一致（不包括root节点）。
- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. Defaults to True.

- Returns:
    - int: 0表示发送成功，否则发送失败。

**collect(tag: str = '@', use_pickle: bool = True) -> List[Any]**

中心节点收集数据

- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. Defaults to True.

- Returns:
    - List[Any]: 收到的数据。list长度等于其他通信节点的数量，第i个位置的数据为第i个通信节点发送的数据。通信节点的顺序与初始化时ids中的节点顺序一致（不包括root节点）。


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

非中心节点发送数据到中心节点

- value (Any): 发送的数据, 任意类型。
- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据序列化. 若数据已序列化则可设为False, 否则应为True. Defaults to True.

- Returns: 
    - int: 0表示发送成功，否则发送失败。


**recv(tag: str = '@', use_pickle: bool = True) -> Any**

非中心节点从中心节点接收数据

- tag (str, optional): 若auto_offset为False，则应手动输入tag并确保不同轮次通信的tag不重复. Defaults to '@'.
- use_pickle (bool, optional): 是否使用pickler方法将数据反序列化，应与发送方参数设置相同. Defaults to True.

- Returns: 
    - Any: 收到的数据.
    

聚合模块
===========

中心节点
-----------

**get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]**

创建中心节点聚合模块实例

- sec_conf (dict): 安全参数。包含关键字method, 值为plain或者otp. 若为otp，则还应包含otp的配置参数，具体的参数见下方示例。
- root_id (str, optional): 中心节点id. 默认取assist_trainer的id. Defaults to ''.
- leaf_ids (list[str], optional): 非中心节点id列表. 默认取除label_trainer和trainer的并集. Defaults to [].

- Returns:
    - Union[AggregationPlainRoot, AggregationOTPRoot]: 根据sec_conf的配置返回AggregationPlainRoot或AggregationOTPRoot的实例。

sec_conf的示例如下：

*明文的配置：*

.. code-block:: json

    {
        "method": "plain"
    }
    
*一次一密的配置：*

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

**set_initial_params(params: OrderedDict) -> None**

初始化待聚合数据

- params (OrderedDict): 原始全局数据.


**aggregate() -> OrderedDict**

接收非中心节点数据并计算聚合数据

- Returns:
    - OrderedDict: 聚合后的数据.


**broadcast(params: OrderedDict) -> int:**

广播数据

- params (OrderedDict): 待分发的全局数据.

- Returns:
    - int: 0表示广播成功，否则失败.


非中心节点
-----------

**get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]**

创建非中心节点聚合模块实例

- sec_conf (dict): 安全参数。与get_aggregation_leaf_inst中sec_conf参数相同。
- root_id (str, optional): 中心节点id. 默认取assist_trainer的id. Defaults to ''.
- leaf_ids (list[str], optional): 非中心节点id列表. 默认取除label_trainer和trainer的并集. Defaults to [].

- Returns:
    - Union[AggregationPlainLeaf, AggregationOTPLeaf]: 根据sec_conf的配置返回AggregationPlainLeaf或AggregationOTPLeaf的实例。


**upload(parameters: OrderedDict, parameters_weight: float) -> int**

上传本地数据到中心节点

- parameters (OrderedDict): 要上传的数据.
- parameters_weight (float): 上传数据的权重.

- Returns:
    - int: 0表示上传成功，否则失败.


**download() -> OrderedDict**

从中心节点下载数据

- Returns:
    - OrderedDict: 下载数据.

