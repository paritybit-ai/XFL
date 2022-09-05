# Copyright 2022 The XFL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import ray

from algorithm.core.data_io import NdarrayIterator
from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.tree_structure import Node, NodeDict
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.crypto.paillier.utils import get_core_num
from common.utils.logger import logger
from common.utils.utils import save_model_config
from service.fed_config import FedConfig
from .base import VerticalXgboostBase
from .decision_tree_trainer import VerticalDecisionTreeTrainer


class VerticalXgboostTrainer(VerticalXgboostBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        super().__init__(train_conf, is_label_trainer=False, *args, **kwargs)
        self.channels = dict()
        self.channels["encryption_context"] = BroadcastChannel(
            name="encryption_context")
        self.channels["individual_grad_hess"] = BroadcastChannel(
            name="individual_grad_hess")
        self.channels["tree_node"] = BroadcastChannel(name="tree_node")

        self.channels["summed_grad_hess"] = DualChannel(name="summed_grad_hess_" + FedConfig.node_id,
                                                        ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
        self.channels["min_split_info"] = DualChannel(name="min_split_info_" + FedConfig.node_id,
                                                      ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
        self.channels["sample_index_after_split"] = DualChannel(name="sample_index_after_split_" + FedConfig.node_id,
                                                                ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
        self.channels["val_com"] = DualChannel(name="val_com_" + FedConfig.node_id,
                                               ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
        self.channels["restart_com"] = DualChannel(name="restart_com_" + FedConfig.node_id,
                                                   ids=FedConfig.get_label_trainer() + [FedConfig.node_id])
        self.channels["early_stop_com"] = DualChannel(name="early_stop_com_" + FedConfig.node_id,
                                                      ids=FedConfig.get_label_trainer() + [FedConfig.node_id])

        if isinstance(self.xgb_config.encryption_param, (PlainParam, type(None))):
            self.public_context = None
        elif isinstance(self.xgb_config.encryption_param, PaillierParam):
            self.public_context = self.channels["encryption_context"].recv(
                use_pickle=False)
            self.public_context = Paillier.context_from(self.public_context)
        else:
            raise TypeError(
                f"Encryption param type {type(self.xgb_config.encryption_param)} not valid.")
        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.identity,
            "filename": self.output.get("model", {"name": "vertical_xgboost_host.json"})["name"]
        }]

        ray.init(num_cpus=get_core_num(self.xgb_config.max_num_cores),
                 ignore_reinit_error=True)

    def fit(self):
        # nodes_dict = {}
        node_dict = NodeDict()

        for tree_idx in range(self.xgb_config.num_trees):
            # training section
            logger.info("Tree {} start training.".format(tree_idx))

            restart_status = 0
            while True:
                sampled_features, feature_id_mapping = self.col_sample()
                # col index in feature
                cat_columns_after_sampling = list(
                    filter(lambda x: feature_id_mapping[x] in self.cat_columns, list(feature_id_mapping.keys())))
                split_points_after_sampling = [self.split_points[feature_id_mapping[k]] for k in
                                               feature_id_mapping.keys()]

                trainer = VerticalDecisionTreeTrainer(tree_param=self.xgb_config,
                                                      features=sampled_features,
                                                      cat_columns=cat_columns_after_sampling,
                                                      split_points=split_points_after_sampling,
                                                      channels=self.channels,
                                                      encryption_context=self.public_context,
                                                      feature_id_mapping=feature_id_mapping,
                                                      tree_index=tree_idx)
                nodes = trainer.fit()

                restart_status = self.channels["restart_com"].recv()

                if restart_status != 1:
                    break

                logger.info(f"trainer tree {tree_idx} training restart.")

            logger.info("Tree {} training done.".format(tree_idx))
            if restart_status == 2:
                logger.info(
                    "trainer early stopped. because a tree's root is leaf.")
                break

            node_dict.update(nodes)

            if self.xgb_config.run_goss:
                self.predict_on_tree(nodes, self.train_dataset)

            # valid section
            logger.info(
                "trainer: Validation on tree {} start.".format(tree_idx))
            self.predict_on_tree(nodes, self.val_dataset)

            if self.channels["early_stop_com"].recv():
                logger.info("trainer early stopped.")
                break
            logger.info("Validation on tree {} done.".format(tree_idx))

            if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get(
                    "save_frequency") == 0:
                self.save(node_dict, epoch=tree_idx + 1)

        # model preserve
        self.save(node_dict, final=True)
        ray.shutdown()

    def _make_indicator_for_prediction(self, nodes: Dict[str, Node], feature: np.ndarray):
        indicator = {}
        for node_id, node in nodes.items():
            feature_idx = node.split_info.feature_idx
            data = feature[:, feature_idx]
            if node.split_info.is_category:
                indicator[node_id] = np.isin(data, node.split_info.left_cat)
            else:
                indicator[node_id] = (data < node.split_info.split_point)
        return indicator

    def predict_on_tree(self, nodes: Dict[str, Node], data_iterator: NdarrayIterator):
        for data in data_iterator:
            indicator = self._make_indicator_for_prediction(nodes, data)
            indicator = {k: np.packbits(v) for k, v in indicator.items()}
            self.channels["val_com"].send(indicator)

    def predict_on_boosting_tree(self, nodes: Dict[str, Node], data_iterator: NdarrayIterator):
        self.predict_on_tree(nodes, data_iterator)

    def save(self, node_dict: NodeDict, epoch: int = None, final: bool = False):
        if final:
            save_model_config(stage_model_config=self.export_conf, save_path=Path(
                self.output.get("model")["path"]))

        save_dir = str(Path(self.output.get("model")["path"]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name_list = self.output.get("model")["name"].split(".")
        name_prefix, name_postfix = ".".join(
            model_name_list[:-1]), model_name_list[-1]
        if not final and epoch:
            model_name = name_prefix + "_{}".format(epoch) + "." + name_postfix
        else:
            model_name = name_prefix + "." + name_postfix
        model_path = os.path.join(save_dir, model_name)

        xgb_output = node_dict.to_dict()
        xgb_output = {k: v for k, v in sorted(xgb_output.items())}

        with open(model_path, 'w') as f:
            json.dump(xgb_output, f)
        logger.info("model saved as: {}.".format(model_path))

    def load_model(self):
        model_path = Path(
            self.input.get("pretrain_model", {}).get("path", ''),
            self.input.get("pretrain_model", {}).get("name", '')
        )
        with open(model_path, 'rb') as f:
            json_dict = json.load(f)

        node_dict = NodeDict.from_dict(json_dict)
        return node_dict

    def check_dataset(self):
        self.channels["check_dataset_com"] = BroadcastChannel(
            name="check_dataset_com")
        n = len(self.test_dataset)
        self.channels["check_dataset_com"].send(n)

    def predict(self):
        self.check_dataset()
        node_dict = self.load_model()
        self.predict_on_boosting_tree(nodes=node_dict.nodes,
                                      data_iterator=self.test_dataset)
