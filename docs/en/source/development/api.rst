====
API
====

Communication Module
=====================

Point-to-point communication
---------------------------------

**class DualChannel(name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True)**

Creates point-to-point communication instance
    
- name (str): channel name.
- ids (list): id list for both parties of communication.
- job_id (Union[str, int], optional): id of the federated learning task, will be obtained interiorly if it is set to "".
- auto_offset (bool, optional): whether to accumulate communication rounds automatically. When setting to false, the tag should be manually entered before calling a specific communication method while ensuring that different tags are used in different rounds. Default: True.


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Send data

- value (Any): data to send. Any type.
- tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

- Returns: 
    - int: 0 means success in sending, otherwise failure.


**recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any**

Receive data

- tag (str, optional): if auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to deserialize data with pickle. It should be identical to the sender's setting. Default: True.
- wait (bool, optional): wheter to wait for receiving to complete. If set to false, return immediately. Default: True.

- Returns: 
    - Any: if wait is set to true, return the sender's data of same round or same tag. If wait is set to false, return the recieved data after complete data has been recieved or None otherwise.


**swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any**

Swap data

- value (Any): data to send. Any type.
- tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to use pickle for data serialization and deserialization. Default: True.

- Returns:
    - Any: Data from the other party


Broadcast Communication
------------------------

**class BroadcastChannel(name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True)**
    
Create broadcast channel instance

- name (str): channel name.
- ids (List[str], optional): id list of all communication parties, defaults to retrieve ids of all parties. Default: [].
- root_id (str, optional): root node id of broadcast channel, as which the id of label trainer by default is obtained. Default: ''.
- job_id (Union[str, int], optional): id of the federated learning task, will be obtained interiorly if it is set to "".
- auto_offset (bool, optional): whether to accumulate communication rounds automatically. When setting to false, the tag should be manually entered before calling a specific communication method while ensuring that different tags are used in different rounds. Default: True.


**broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Broadcast data from root node

- value (Any): data to broadcast. Any type.
- tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

- Returns:
    - int: 0 means success in sending, otherwise failure.

**scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int**

Scatter data by root node (different data for different nodes)

- values (List[Any]): data to scatter. The length of the list should equal the number of leaf nodes. The i-th data is sent to the i-th node. The order of the communication nodes is the same as that of the nodes in the ids at initialization (excluding root node).
- tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

- Returns:
    - int: 0 means success in sending, otherwise failure.

**collect(tag: str = '@', use_pickle: bool = True) -> List[Any]**

Collect data by root node

- tag (str, optional): if auto_offset is false, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults: True.

- Returns:
    - List[Any]: collected data. The length of the list should equal the number of leaf nodes. The i-th data is sent to the i-th node. The order of the communication nodes is the same as that of the nodes in the ids at initialization (excluding root node).


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Send data to root node from leaf node

- value (Any): data to send, Any type.
- tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

- Returns: 
    - int: 0 means success in sending, otherwise failure.


**recv(tag: str = '@', use_pickle: bool = True) -> Any**

Receive data from root node by leaf node

- tag (str, optional): if auto_offset is false, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

- Returns: 
    - Any: data received.
    

Aggregation Module
======================

Root Node
-----------

**get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]**

Create root node instance

- sec_conf (dict): security configuration. Detailed configurations are shown as below.
- root_id (str, optional): id of root node. it will be set to assister_trainer by default. Default: ''.
- leaf_ids (list[str], optional): id list of leaf nodes. By default it will be set to the union of label_trainer and trainer. Default: [].

- Returns:
    - Union[AggregationPlainRoot, AggregationOTPRoot]: instance of AggregationPlainRoot or AggregationOTPRoot.


Example of sec_conf:

**Configuration for plain aggregation**

.. code-block:: json

    {
        "method": "plain"
    }
    
**Configuration for otp aggregation**

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

Set initial parameters to send by root node

- params (OrderedDict): initial parameters of model.


**aggregate() -> OrderedDict**

Receive data from leaf nodes and aggregate

- Returns:
    - OrderedDict: result after aggregation.


**broadcast(params: OrderedDict) -> int:**

Broadcast data to all the leaf nodes

- params (OrderedDict): data to broadcast.

- Returns:
    - int: 0 means success in sending, otherwise failure.


Leaf Node
-----------

**get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]**

Create leaf node instance

- sec_conf (dict): security configuration. The same with the security configuration of get_aggregation_root_inst.
- root_id (str, optional): id of root node. it will be set to assister_trainer by default. Default: ''.
- leaf_ids (list[str], optional): id list of leaf nodes. By default it will be set to the union of label_trainer and trainer. Default: [].

- Returns:
    - Union[AggregationPlainLeaf, AggregationOTPLeaf]: instance of AggregationPlainLeaf or AggregationOTPLeaf.


**upload(parameters: OrderedDict, parameters_weight: float) -> int**

Upload data and data's weight to root node

- parameters (OrderedDict): data to upload.
- parameters_weight (float): weight of uploading data.

- Returns:
    - int: 0 means success in sending, otherwise failure.


**download() -> OrderedDict**

Download data from root node

- Returns:
    - OrderedDict: downloaded data.

