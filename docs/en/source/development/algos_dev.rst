=======================
Operator Development
=======================

Create Operator File Structure
=================================

Before creating an operator in XFL, you need to first create the directory and files where the operator will be located. 
The built-in operators in XFL are stored in the `python/alogorithm/framework` folder.
To create a new operator, first, create the operator directory in consistent with the form `federation type` [#type]_ / `algorithm name`, 
second, create the corresponding `.py` file according to the `federation role` [#role]_ that the operator will involve with.
Take the creation of Vertical Xgboost operator as an example. The federation type is vertical and the algorithm name is xgboost. 
The operator includes two roles: label_trainer and trainer. Therefore, the file structure should be as follows:

::

    | — vertical
    | | — xgboost
    | | | — label_trainer.py
    | | | — trainer.py


Create Operator Class
=======================

After creating the operator files, you need to create corresponding operator classes for each federated role. 
XFL supports automatic discovery of operators, which requires the developers to follow the following naming rules to name the class of operator.
The operator name is obtained by the following steps:

1. Concatenate the federation type, algorithm name, and federation role by underscore '_';
#. Switch the first letter and the letters right after the underscores to uppercase;
#. Remove the underscores to get the class name.

Taking Vertical Xgboost operator as an example, you need to create class VerticalXgboostLabelTrainer in 'label_trainer.py', and class VerticalXgboostTrainer in 'trainer.py'.
All operators accept a dictionary type parameter 'train_info'. Each class should implement a function called 'fit', 
where the training process is expected to be implemented in. For example:

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


Develop of Operator
======================

Parameters of Operator
------------------------


The operator takes 'train_conf' as input, which is of type dict. The content of train_conf is read from the configuration(json format) of this operator for each party. 
The major parameters define in 'train_conf' is as follows:

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

In the above, identity specifies the role of party who call the operator. It should be one of `label_trainer`, `trainer`, and `assist_trainer`. 
`model_info.name` is the name of the operator which consists of operator type (horizontal, vertical or local) and algorithm name concatenated by a underscore,
`model_info.config` defines the construction of the model(if it is needed).
`train_info` is supposed to contain parameters for training.

Structure of Operator
----------------------

An operator should contain at least two functions, __init__ and fit. XLF initializes the operator by __init__, and trains the operator by calling fit function. 
We suggest to put the code that called only once into __init__ function, such as initialization of dataset, model, loss function, metric, optimizer, and communication channel.


Tools for Development
======================

Communication module
---------------------

XFL encapsulates a concise communication module based on grpc and redis. This communication module provides two modes of communication: 
point-to-point communication and broadcast communication. Developers can create channels, send and receive data by using this module.

1. Point-to-point communication

- Create channel

.. code-block:: python

    class DualChannel(name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True):
    
        """
        Args:
            name (str): channel name.
            ids (list): id list for both parties of communication.
            job_id (Union[str, int], optional): id of the federated learning task, will be obtained interiorly if it is set to "".
            auto_offset (bool, optional): whether to accumulate communication rounds automatically. When setting to false, the tag should be manually entered before calling a specific communication method while ensuring that different tags are used in different rounds. Default: True.
        """
   

- Send data

.. code-block:: python

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """"
        Args:
            value (Any): data to send. Any type.
            tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

        Returns: 
            int: 0 means success in sending, otherwise failure.
        """

- Receive data

.. code-block:: python

    recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any:
        """
        Args:
            tag (str, optional): if auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to deserialize data with pickle. It should be identical to the sender's setting. Default: True.
            wait (bool, optional): wheter to wait for receiving to complete. If set to false, return immediately. Default: True.

        Returns: 
            Any: if wait is set to true, return the sender's data of same round or same tag. If wait is set to false, return the recieved data after complete data has been recieved or None otherwise.
        """
        
- Swap data

.. code-block:: python

    swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any:

        """
        Args:
            value (Any): data to send. Any type.
            tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to use pickle for data serialization and deserialization. Default: True.

        Returns:
            Any: Data from the other party
        """


:Example:

Assume there is only one label trainer and one trainer in the federated task.

- trainer

.. code-block:: python

    from common.communication.gRPC.python.channel import DualChannel
    from service.fed_config import FedConfig

    demo_chann = DualChannel(name="demo_dual_chann", ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
    demo_chann.send(1)
    b = demo_chann.swap(2) 
    # b = 3


- label trainer

.. code-block:: python

    from common.communication.gRPC.python.channel import DualChannel
    from service.fed_config import FedConfig

    demo_chann = DualChannel(name="demo_dual_chann", ids=[FedConfig.node_id] + FedConfig.get_trainer())
    a = demo_chann.recv() 
    # a = 1
    b = demo_chann.swap(3) 
    # b = 2


2. Broadcast communication

- Create channel

.. code-block:: python

    class BroadcastChannel(name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True):
    
        """
            name (str): channel name.
            ids (List[str], optional): id list of all communication parties, defaults to retrieve ids of all parties. Default: [].
            root_id (str, optional): root node id of broadcast channel, as which the id of label trainer by default is obtained. Default: ''.
            job_id (Union[str, int], optional): id of the federated learning task, will be obtained interiorly if it is set to "".
            auto_offset (bool, optional): whether to accumulate communication rounds automatically. When setting to false, the tag should be manually entered before calling a specific communication method while ensuring that different tags are used in different rounds. Default: True.
        """

-  Broadcast data from root node

.. code-block:: python

    broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """
            Args:
                value (Any): data to broadcast. Any type.
                tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
                use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

            Returns:
                int: 0 means success in sending, otherwise failure.
        """


- Scatter data by root node (different data for different nodes)

.. code-block:: python

    scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int:

        """
        Args:
            values (List[Any]): data to scatter. The length of the list should equal to the number of non-root nodes. The i-th data is sent to the i-th node. The order of the communication nodes is the same as that of the nodes in the ids at initialization (excluding root node).
            tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

        Returns:
            int: 0 means success in sending, otherwise failure.
        """

- Collect data by root node

.. code-block:: python

    collect(tag: str = '@', use_pickle: bool = True) -> List[Any]:

        """
        Args:
            tag (str, optional): if auto_offset is false, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Defaults: True.

        Returns:
            List[Any]: collected data. The length of the list equals to the number of non-root nodes. The i-th data is sent to the i-th node. The order of the communication nodes is the same as that of the nodes in the ids at initialization (excluding root node).
        """

- Send data to root node from leaf node

.. code-block:: python

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """
        Args:
            value (Any): data to send, Any type.
            tag (str, optional): if auto_offset is False, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

        Returns: 
            int: 0 means success in sending, otherwise failure.
        """

- Receive data from root node by leaf node

.. code-block:: python

    recv(tag: str = '@', use_pickle: bool = True) -> Any:

        """
        Args:
            tag (str, optional): if auto_offset is false, the tag need to be mannually entered while ensuring that different tags are used in different rounds. Default: '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. Default: True.

        Returns: 
            Any: data received.
        """


:Example:

Assume assist_trainer is the root node while leaf nodes include two trainers: node-1 and node-2.

- assist_trainer

.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

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


Aggregation Module
--------------------

There are two types of participants in the aggregation module: root and leaf. Root is the center node, which can broadcast and aggregate parameters. 
Leaf is non-center node, which can upload and download parameters. 
The aggregation module supports plain aggregation and encrypted aggregation. The encrypted aggregation supports one time pad (OTP [#FedAvg]_ ) encryption.

1. Root node

XFL supports two types of root node initialization: AggregationPlainRoot and AggregationOTPRoot. AggregationOTPRoot supports OTP encryption.

- Create instance

.. code-block:: python

    get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]:

        """
        Args:
            sec_conf (dict): security configuration. Detailed configurations are shown as below.
            root_id (str, optional): id of root node. it will be set to assister_trainer by default. Default: ''.
            leaf_ids (list[str], optional): id list of leaf nodes. By default it will be set to the union of label_trainer and trainer. Default: [].

        Returns:
            Union[AggregationPlainRoot, AggregationOTPRoot]: instance of AggregationPlainRoot or AggregationOTPRoot.
        """

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

Methods bound to root node: 

- set initial parameters to send by root node

.. code-block:: python

    set_initial_params(params: OrderedDict) -> None:

        """
        Args:
            params (OrderedDict): initial parameters of model.
        """

- receive data from leaf nodes and aggregate with the formula: :math:`\sum_{i} parameters_i \cdot parameters\_weight_i`

.. code-block:: python

    aggregate() -> OrderedDict:

        """
        Returns:
            OrderedDict: result after aggregation.
        """

- broadcast data to all the leaf nodes

.. code-block:: python

    broadcast(params: OrderedDict) -> int:

        """
        Args:
            params (OrderedDict): data to broadcast.

        Returns:
            int: 0 means success in sending，otherwise failure.
        """

2. leaf node

Corresponds with the root node, there are also two types of leaf node instance: AggregationPlainLeaf and AggregationOTPLeaf. The initialization function is as follows:

- Create instance

.. code-block:: python

    get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]:

        """
        Args:
            sec_conf (dict): security configuration. The same with the security configuration of get_aggregation_root_inst.
            root_id (str, optional): id of root node. it will be set to assister_trainer by default. Default: ''.
            leaf_ids (list[str], optional): id list of leaf nodes. By default it will be set to the union of label_trainer and trainer. Default: [].

        Returns:
            Union[AggregationPlainLeaf, AggregationOTPLeaf]: instance of AggregationPlainLeaf or AggregationOTPLeaf.
        """

Methods bound to leaf node:

- Upload data and data's weight to root node

.. code-block:: python

    upload(parameters: OrderedDict, parameters_weight: float) -> int:

        """
        Args:
            parameters (OrderedDict): data to upload.
            parameters_weight (float): weight of uploading data.

        Returns:
            int: 0 means success in sending, otherwise failure.
        """

- Download data from root node

.. code-block:: python

    download() -> OrderedDict:

        """
        Returns:
            OrderedDict: downloaded data.
        """




Develop Horizontal Operator
=============================

Differ from the diversity of the communication patterns in vertical federation learning,
the communication model of horizontal federation is universal.
XFL provides several preseted template classes, which can be inherited by custom classes to develop new horizontal operators. 
For example, the FedAvg template contains `FedAvgTemplateAssistTrainer` (`python/algorithm/core/horizontal/template/torch/fedavg/assist_trainer.py`) 
and `FedAvgTemplateLabelTrainer` (`python/algorithm/core/horizontal/template/torch/fedavg/label_trainer.py`).
An example of developing an operator using this template can be found at `python/algorithm/framework/horizontal/logistic_regression/assist_trainer.py`, 
`python/algorithm/framework/horizontal/logistic_regression/label_trainer.py`.

:Notes:

.. [#type] XFL supports three types of operators for the moment: horizontal, vertical, and local.
.. [#role] XFL supports three types of role in federated learning: assist_trainer, label_trainer, and trainer.
.. [#FedAvg] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]//proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 1175-1191.
