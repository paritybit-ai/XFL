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
- ids (list): id list for the two parties.
- job_id (Union[str, int], optional): id of federated learning taks，retrieved automatically by default. Defaults to "" .
- auto_offset (bool, optional): if accumulate automatically communication numbers. When setting to False, tag should be manually entered during communication and it is mandatory to ensure that different tags are used in different rounds. Defaults to True.


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Send data

- value (Any): data to send, arbitrary type.
- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns: 
    - int: 0 means success in sending，otherwise failure.


**recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any**

Receive data

- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to deserialize data with pickle. It should be identical to the sender's parameter. Defaults to True.
- wait (bool, optional): wheter to wait for receiving to complete. If set to False, return immediately. Defaults to True.

- Returns: 
    - Any: If wait is set to True, return the data of the same round or the same tag from sender. If wait is set to False, return the data when receiving is complete or Nono otherwise.


**swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any**

Swap data

- value (Any): data to send, Any type.
- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): wheter to use pickle for data serialization and deserialization. Defaults to True.

- Returns:
    - Any: data from the other party


Broadcast Communication
------------------------

**class BroadcastChannel(name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True)**
    
Create broadcast channel instance

- name (str): channel name.
- ids (List[str], optional): id list of all parties, defautls to retrieve ids of all parties. Defaults to [].
- root_id (str, optional): root node id of broadcast channel, retrieve the id of label trainer by default. Defaults to ''.
- job_id (Union[str, int], optional): id of federated learning taks，retrieved automatically by default. Defaults to "".
- auto_offset (bool, optional): if accumulate automatically communication numbers. When setting to False, tag should be manually entered during communication and it is mandatory to ensure that different tags are used in different rounds. Defaults to True.

**broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Broadcast data from root node

- value (Any): data to broadcast. Any type.
- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns:
    - int: 0 means success in sending，otherwise failure.

**scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int**

Scatter data by root node (different data for different nodes)

- values (List[Any]): data to scatter. The length of the list should equal the number of non-root nodes. The i-th data is sent to the i-th node. The order of noda and data is that when initializing nodes (excluding root node).
- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns:
    - int: 0 means success in sending，otherwise failure.

**collect(tag: str = '@', use_pickle: bool = True) -> List[Any]**

Collect data by root node

- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns:
    - List[Any]: received data.The length of the list should equal the number of non-root nodes. The i-th data is sent to the i-th node. The order of noda and data is that when initializing nodes (excluding root node).


**send(value: Any, tag: str = '@', use_pickle: bool = True) -> int**

Send data to root node from leaf node

- value (Any): data to send, Any type.
- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns: 
    - int: 0 means success in sending，otherwise failure.


**recv(tag: str = '@', use_pickle: bool = True) -> Any**

Receive data from root node by leaf node

- tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
- use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults to True.

- Returns: 
    - Any: data received
    

Aggregation Module
======================

Root Node
-----------

**get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]**

Create root node instance

- sec_conf (dict): configuration of security. Includes the key method, with values 'plain' or 'otp'. If method is 'otp', configuration for opt should also be included. See the example below.
- root_id (str, optional): id of root node. Assister_trainer id by default. Defaults to ''.
- leaf_ids (list[str], optional): id list of leaf node. The union of label_trainer and trainer by default. Defaults to [].

- Returns:
    - Union[AggregationPlainRoot, AggregationOTPRoot]: instance of AggregationPlainRoot or AggregationOTPRoot configured with the sec_conf.

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

- params (OrderedDict): dictionary of initial parameters.


**aggregate() -> OrderedDict**

Receive data from leaf nodes and aggregate

- Returns:
    - OrderedDict: data after aggregation.


**broadcast(params: OrderedDict) -> int:**

Broadcast data to all leaf node

- params (OrderedDict): data to broadcast.

- Returns:
    - int: 0 means success in sending，otherwise failure.


Leaf Node
-----------

**get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]**

Create leaf node instance

- sec_conf (dict): configuration of security. Must be the same with that of get_aggregation_root_inst.
- root_id (str, optional): id of root node. Assister_trainer id by default. Defaults to ''.
- leaf_ids (list[str], optional): id list of leaf node. The union of label_trainer and trainer by default. Defaults to [].

- Returns:
    - Union[AggregationPlainLeaf, AggregationOTPLeaf]: instance of AggregationPlainLeaf or AggregationOTPLeaf configured with sec_conf.


**upload(parameters: OrderedDict, parameters_weight: float) -> int**

Upload data and weight to root node

- parameters (OrderedDict): data to upload.
- parameters_weight (float): weight of uploading data.

- Returns:
    - int: 0 means success in sending，otherwise failure.


**download() -> OrderedDict**

Download data from root node

- Returns:
    - OrderedDict: downloaded data.

