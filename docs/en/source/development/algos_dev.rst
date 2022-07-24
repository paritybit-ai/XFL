=======================
Operator Development
=======================

Create Operator File
=======================

Before creating an operator in XFL, you must first create the directory and files where the operator will be located. The built-in operators in XFL are stored in the `python/alogorithm/framework` folder。
when creating a new operator，you need to create the operator directory in consistent with the form "federation type" [#type]_ /"algorithm name", then you need to create corresponding .py file based on the “federation role” [#role]_ which the operator needs.
Take the creation of Vertical Xgboost operator as an example. The operator federation type is vertical and the algorithm name is xgboost. The operator contains two roles: label_trainer and trainer. Therefore, the created file directory structure should be as follows:

::

    | — vertical
    | | — xgboost
    | | | — label_trainer.py
    | | | — trainer.py


Create Operator Class
=======================

After creating the operator files, you need to create corresponding operator classes for each federated role. XFL supports automatic discovery of operators, which requires the following of naming conventions as below.
The operator name is obtained by the following steps:

1. Join the federation type, operator name, and federation role with underscore'_';
#. Change the initial letter and the letters after the underscores to uppercase;
#. Remove the underscores to get the class name.

Taking Vertical Xgboost operator as an example, you need to create the class VerticalXgboostLabelTrainer in label_trainer.py, and create the class VerticalXgboostTrainer in trainer.py.
All operators accept the same parameter train_info, which is a dictionary type, see [Algorithm Parameter Explanation](section1.4/section1.4.3.md). Each class must contain a fit method which is implemented for the training process of operators and takes no other parameters. For example:

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


Develop Operator
==================

Parameters of Operator
------------------------


The operator receives train_conf as an input parameter, which is of dict type. The content of train_conf should be consistent with the parameters when the user calls the operator (through the API or through the json file). The main information in train_conf is as follows:

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

In the above, identity specifies the role of the operator caller. It should be one of label_trainer, trainer, or assist_trainer. model_info.name is the name of the operator, formed by operator type (horizontal, vertical or local) and algorithm name concatenated by underscore. input contains input data information, output contains output data information, and train_info contains necessary training information.

Structure of Operator
----------------------

The operator shoulc contain __init__ and fit methods. XLF initializes the operator through __init__, and trains the operator through fit. It is recommended to put code for one time operation such as data initialization, model instantiation, loss function, metric, optimizer, and communication channel in __init__ method and code for model training in fit.

Tools for Development
======================

Communication module
---------------------

XFL has a concise communication module as a wrapper around grpc+redis. This communication module provides two modes of communication: point-to-point communication and broadcast communication. Developers can create channels, send and receive data using this module.

1. Point-to-point communication

- Create channel

.. code-block:: python

    class DualChannel(name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True):
    
        """
        Args:
            name (str): channel name.
            ids (list): id list for the two parties.
            job_id (Union[str, int], optional): id of federated learning taks，retrieved automatically by default. Defaults to "".
            auto_offset (bool, optional): if accumulate automatically communication numbers.
                When setting to False, tag should be manually entered during communication and it is mandatory to ensure that different tags are used in different rounds.
                Defaults to True.
        """
   

- Send data

.. code-block:: python

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """"
        Args:
            value (Any): data to send, arbitrary type.
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.

        Returns:
            int: 0 means success in sending，otherwise failure.
        """

- Receive data

.. code-block:: python

    recv(tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any:
        """
        Args:
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to deserialize data with pickle. It should be identical to the sender's parameter. Defaults to True.
                Defaults to True.
            wait (bool, optional): wheter to wait for receiving to complete. If set to False, return immediately. Defaults to True.

        Returns:
            Any: If wait is set to True, return the data of the same round or the same tag from sender. If wait is set to False, return the data when receiving is complete or Nono otherwise.
        """
        
- Swap data

.. code-block:: python

    swap(value: Any, tag: str = '@', use_pickle: bool = True) -> Any:

        """
        Args:
            value (Any): data to send, Any type.
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): wheter to use pickle for data serialization and deserialization. Defaults to True.

        Returns:
            Any: data from the other party
        """


:Example:

Assume thers is one label trainer and one trainer.

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
        Args:
            name (str): channel name.
            ids (List[str], optional): id list of all parties, defautls to retrieve ids of all parties. Defaults to [].
            root_id (str, optional): root node id of broadcast channel, retrieve the id of label trainer by default. Defaults to ''.
            job_id (Union[str, int], optional): id of federated learning taks，retrieved automatically by default. Defaults to "".
            auto_offset (bool, optional): if accumulate automatically communication numbers.
                When setting to False, tag should be manually entered during communication and it is mandatory to ensure that different tags are used in different rounds.
                Defaults to True.
        """

-  Broadcast data from root node

.. code-block:: python

    broadcast(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """
        Args:
            value (Any): data to broadcast. Any type.
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.

        Returns:
            int: 0 means success in sending，otherwise failure.
        """


- Scatter data by root node (different data for different nodes)

.. code-block:: python

    scatter(values: List[Any], tag: str = '@', use_pickle: bool = True) -> int:

        """
        Args:
            values (List[Any]): data to scatter. The length of the list should equal the number of non-root nodes. The i-th data is sent to the i-th node. The order of noda and data is that when initializing nodes (excluding root node).
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.

        Returns:
            int: 0 means success in sending，otherwise failure.
        """

- Collect data by root node

.. code-block:: python

    collect(tag: str = '@', use_pickle: bool = True) -> List[Any]:

        """
        Args:
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.

        Returns:
            List[Any]: received data.The length of the list should equal the number of non-root nodes. The i-th data is sent to the i-th node. The order of noda and data is that when initializing nodes (excluding root node).
        """

- Send data to root node from leaf node

.. code-block:: python

    send(value: Any, tag: str = '@', use_pickle: bool = True) -> int:

        """
        Args:
            value (Any): data to send, Any type.
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.
        Returns:
            int: 0 means success in sending，otherwise failure.
        """

- Receive data from root node by leaf node

.. code-block:: python

    recv(tag: str = '@', use_pickle: bool = True) -> Any:

        """
        Args:
            tag (str, optional): If auto_offset is False, the tag need to be mannually entered and it is mandatory to ensure that different tags are used in different rounds. Defaults to '@'.
            use_pickle (bool, optional): whether to serialize data with pickle. If data is already serialized, it should be set to true, otherwise set to false. 
                Defaults to True.

        Returns:
            Any: data received
        """


:Example:

Assume assist_trainer is the root node, non-root nodes include two trainers: node-1 and node-2.

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

There are two types of participants in the aggregation module: root and leaf. Root is the center node, which can broadcast and aggregate parameters. Leaf is a non-center node, which can upload and download parameters. We will use root/center, lean/non-center interchangeably. The aggregation module supports plain aggregation and encrypted aggregation. The encrypted aggregation supports one-time pad (OTP [#FedAvg]_ ) encryption.

1. Root node

XFL supports two types of root node initialization: AggregationPlainRoot and AggregationOTPRoot. AggregationOTPRoot supports OTP encryption.

- Create instance

.. code-block:: python

    get_aggregation_root_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainRoot, AggregationOTPRoot]:

        """
        Args:
            sec_conf (dict): configuration of security. Includes the key method, with values 'plain' or 'otp'. If method is 'otp', configuration for opt should also be included. See the example below.
            root_id (str, optional): id of root node. Assister_trainer id by default. Defaults to ''.
            leaf_ids (list[str], optional): id list of leaf node. The union of label_trainer and trainer by default. Defaults to [].

        Returns:
            Union[AggregationPlainRoot, AggregationOTPRoot]: instance of AggregationPlainRoot or AggregationOTPRoot configured with the sec_conf.
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

Methods of root node:

- set initial parameters to send by root node

.. code-block:: python

    set_initial_params(params: OrderedDict) -> None:

        """
        Args:
            params (OrderedDict): dictionary of initial parameters.
        """

- receive data from leaf nodes and aggregate with the formula: :math:`\sum_{i} parameters_i \cdot parameters\_weight_i`

.. code-block:: python

    aggregate() -> OrderedDict:

        """
        Returns:
            OrderedDict: data after aggregation.
        """

- broadcast data to all leaf node

.. code-block:: python

    broadcast(params: OrderedDict) -> int:

        """
        Args:
            params (OrderedDict): data to broadcast.

        Returns:
            int: 0 means success in sending，otherwise failure.
        """

2. leaf node

Inline with root node, there are also two types of leaf node: AggregationPlainLeaf and AggregationOTPLeaf. The initialization is as follows:

- Create instance

.. code-block:: python

    get_aggregation_leaf_inst(sec_conf: dict, root_id: str = '', leaf_ids: list[str] = []) -> Union[AggregationPlainLeaf, AggregationOTPLeaf]:

        """
        Args:
            sec_conf (dict): configuration of security. Must be the same with that of get_aggregation_root_inst.
            root_id (str, optional): id of root node. Assister_trainer id by default. Defaults to ''.
            leaf_ids (list[str], optional): id list of leaf node. The union of label_trainer and trainer by default. Defaults to [].

        Returns:
            Union[AggregationPlainLeaf, AggregationOTPLeaf]: instance of AggregationPlainLeaf or AggregationOTPLeaf configured with sec_conf.
        """

Leaf node has the following methods:

- Upload data and weight to root node

.. code-block:: python

    upload(parameters: OrderedDict, parameters_weight: float) -> int:

        """
        Args:
            parameters (OrderedDict): data to upload.
            parameters_weight (float): weight of uploading data.

        Returns:
            int: 0 means success in sending，otherwise failure.
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

Different from its vertical counterpart, horizontal federated learning communication is rather standard. XFL provides preset template classes, which can be leveraged to develop horizontal models conveniently. For the moment, XFL provides template classes based on FedAvg, cf `FedAvgTemplateAssistTrainer <../../../../python/algorithm/core/horizontal/template/torch/fedavg/assist_trainer.py>`_ , 
`FedAvgTemplateLabelTrainer <../../../../python/algorithm/core/horizontal/template/torch/fedavg/label_trainer.py>`_ . An example of developing with this template can be found at `HorizontalLogisticRegressionAssistTrainer <../../../../python/algorithm/framework/horizontal/logistic_regression/assist_trainer.py>`_ , `HorizontalLogisticRegressionLabelTrainer <../../../../python/algorithm/framework/horizontal/logistic_regression/label_trainer.py>`_ .


:Notes:

.. [#type] XFL supports three types of operators for the moment: horizontal, vertical, and local.
.. [#role] XFL supports three types of role in federated learning: assist_trainer, label_trainer, and trainer.
.. [#FedAvg] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]//proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 1175-1191.
