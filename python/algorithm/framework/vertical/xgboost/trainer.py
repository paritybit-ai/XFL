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

import pandas as pd
import numpy as np
import ray

from algorithm.core.data_io import NdarrayIterator
from algorithm.core.encryption_param import PaillierParam, PlainParam
from algorithm.core.tree.tree_structure import Node, NodeDict
from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel
from common.crypto.paillier.paillier import Paillier
from common.crypto.paillier.utils import get_core_num
from common.utils.logger import logger
from common.utils.utils import save_model_config, update_dict
from service.fed_config import FedConfig
from .base import VerticalXgboostBase
from .decision_tree_trainer import VerticalDecisionTreeTrainer
from service.fed_job import FedJob
from service.fed_node import FedNode
from common.utils.model_io import ModelIO


class VerticalXgboostTrainer(VerticalXgboostBase):
    def __init__(self, train_conf: dict, *args, **kwargs):
        self.channels = dict()
        self.channels["sync"] = BroadcastChannel(name="sync")
        conf = self._sync_config()
        update_dict(train_conf, conf)
        super().__init__(train_conf, is_label_trainer=False, *args, **kwargs)
        
        self.channels["encryption_context"] = BroadcastChannel(
            name="encryption_context")
        self.channels["individual_grad_hess"] = BroadcastChannel(
            name="individual_grad_hess")
        self.channels["tree_node"] = BroadcastChannel(name="tree_node")
        self.channels["check_dataset_com"] = BroadcastChannel(
            name="check_dataset_com")

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

        if self.train_features is not None:
            input_schema = ','.join([_ for _ in self.train_features.columns if _ not in set(["y", "id"])])
        else:
            input_schema = ""

        self.export_conf = [{
            "class_name": "VerticalXGBooster",
            "identity": self.identity,
            "filename": self.output.get("proto_model", {}).get("name", ''),
            "input_schema": input_schema,
        }]

        ray.init(num_cpus=get_core_num(self.xgb_config.max_num_cores),
                 ignore_reinit_error=True)
        
    def _sync_config(self):
        config = self.channels["sync"].recv()
        return config

    def fit(self):
        self.channels["sync"].send({FedNode.node_name: self.train_names})
        self.check_dataset()
        # nodes_dict = {}
        node_dict = NodeDict()

        for tree_idx in range(1, self.xgb_config.num_trees+1):
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

            # if self.interaction_params.get("save_frequency") > 0 and (tree_idx + 1) % self.interaction_params.get(
            #         "save_frequency") == 0:
            #     self.save(node_dict, epoch=tree_idx + 1)
            
            if self.interaction_params.get("save_frequency") > 0 and tree_idx % self.interaction_params.get(
                    "save_frequency") == 0:
                self.save(node_dict, epoch=tree_idx)

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
                indicator[node_id] = (data <= node.split_info.split_point)
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
            save_model_config(stage_model_config=self.export_conf, save_path=self.output.get("path"))
        
        save_dir = self.output.get("path")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        model_name = self.output.get("model", {}).get("name")
        proto_name = self.output.get("proto_model", {}).get("name")
        
        if model_name:
            out_dict = node_dict.to_dict()
            model_dict = {"nodes": out_dict}
            ModelIO.save_json_model(model_dict, save_dir, model_name, epoch=epoch, version='1.4.0')

        if proto_name:
            model_name_list = self.output.get("proto_model")["name"].split(".")
            name_prefix, name_postfix = ".".join(
                model_name_list[:-1]), model_name_list[-1]
            if not final and epoch:
                model_name = name_prefix + "_epoch_{}".format(epoch) + "." + name_postfix
            else:
                model_name = name_prefix + "." + name_postfix
            model_path = os.path.join(save_dir, model_name)

            xgb_output = node_dict.to_proto()

            with open(model_path, 'wb') as f:
                f.write(xgb_output)
            logger.info("model saved as: {}.".format(model_path))

    def load_model(self):
        pretrain_path = self.input.get("pretrained_model", {}).get("path", '')
        model_name = self.input.get("pretrained_model", {}).get("name", '')
        
        model_path = Path(
            pretrain_path, model_name
        )
        suffix = model_name.split(".")[-1]
        
        if suffix != "pmodel":
            model_dict = ModelIO.load_json_model(model_path)
            node_dict = NodeDict.from_dict(model_dict["nodes"])
        else:
            with open(model_path, 'rb') as f:
                byte_str = f.read()

            node_dict = NodeDict.from_proto(byte_str)

        return node_dict

    def check_dataset(self):
        d = dict()
        if self.train_dataset is not None:
            d["train"] = len(self.train_ids), len(self.train_features.columns)
        if self.val_dataset is not None:
            d["valid"] = len(self.val_ids), len(self.val_features.columns)
        if self.test_dataset is not None:
            d["test"] = len(self.test_ids), len(self.test_features.columns)
        self.channels["check_dataset_com"].send(d)

    def predict(self):
        out_dict = {key: None for key, value in self.train_conf.get("output", {}).items() if key != "path" and value.get("name")}
        self.channels["sync"].send(out_dict)
        self.check_dataset()
        node_dict = self.load_model()
        
        self.predict_on_boosting_tree(nodes=node_dict.nodes,
                                      data_iterator=self.test_dataset)
        
        out_dict = self.channels["sync"].recv()
        
        save_path = self.output.get("path", '')
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            for key in out_dict:
                file_path = Path(save_path, self.output[key]["name"])
                if key == "testset" and file_path:
                    logger.info("predicted results saved at {}".format(file_path))
                    pd.DataFrame({"id": self.test_ids, "pred": out_dict[key]}).to_csv(
                        file_path, float_format="%.6g", index=False, header=True
                    )

        
