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


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import dgl
from dgllife.model import GCNPredictor
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

from algorithm.core.data_io import CsvReader, NpzReader
from common.utils.logger import logger
from algorithm.core.horizontal.template.torch.base import BaseTrainer
from common.utils.config_sync import ConfigSynchronizer
from common.checker.x_types import All
from common.evaluation.metrics import CommonMetrics


class SmilesDataset(Dataset):
    def __init__(self, smiles, graphs, labels):
        assert len(smiles) == len(
            graphs), "Inconsistent lengths of smiles and graphs"
        assert len(graphs) == len(
            labels), "Inconsistent lengths of graphs and labels"

        self.smiles = smiles
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return self.smiles[index], self.graphs[index], self.labels[index]


def collate_molgraphs(data):
    """Function from dgllife.examples.property_prediction.moleculenet.utils
    Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


class Common(BaseTrainer):
    def __init__(self, train_conf: dict) -> None:
        sync_rule = {
            "model_info": All(),
            "train_info": {
                "interaction_params": All(),
                "train_params": {
                    "global_epoch": All(),
                    "aggregation": All(),
                    "encryption": All(),
                    "optimizer": All(),
                    "lr_scheduler": All(),
                    "lossfunc": All(),
                    "metric": All(),
                    "early_stopping": All()
                }
            }
        }
        train_conf = ConfigSynchronizer(train_conf).sync(sync_rule)
        super().__init__(train_conf)

    def _set_model(self) -> nn.Module:
        model_config = self.common_config.model_info.get("config")
        model_params = self._prepare_model_params(model_config)
        model = GCNPredictor(**model_params)
        return model

    def _prepare_model_params(self, model_config):
        config = {}
        config['in_feats'] = model_config.get("input_dim", 100)
        num_gnn_layers = model_config.get('num_gnn_layers', 1)
        config['hidden_feats'] = [model_config.get(
            'gnn_hidden_feats', 64)] * num_gnn_layers
        if model_config['activation'] == 'relu':
            config['activation'] = [F.relu] * num_gnn_layers
        elif model_config['activation'] == 'tanh':
            config['activation'] = [F.tanh] * num_gnn_layers
        else:
            logger.info(f"Setting gnn activation to relu")
            config['activation'] = [F.relu] * num_gnn_layers

        config['dropout'] = [model_config.get("dropout", 0.5)] * num_gnn_layers
        config['batchnorm'] = [model_config.get(
            'batchnorm', False)] * num_gnn_layers
        config['residual'] = [model_config.get(
            "residual", False)] * num_gnn_layers
        config['predictor_hidden_feats'] = model_config.get(
            'predictor_hidden_dim', 64)
        config['n_tasks'] = model_config.get('n_tasks', 1)

        return config

    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None

        conf = input_dataset[0]

        if conf["type"] == "csv":
            path = os.path.join(conf['path'], conf['name'])
            has_label = conf["has_label"]
            has_id = conf['has_id']
            return CsvReader(path, has_id, has_label)
        elif conf["type"] == "npz":
            path = os.path.join(conf['path'], conf['name'])
            return NpzReader(path)
        else:
            return None

    def _set_train_dataloader(self):
        train_data = self._read_data(self.common_config.input_trainset)
        trainset = None
        train_dataloader = None
        if train_data is None:
            return train_dataloader

        # construct smiles
        smiles = train_data.features(type="series").values.reshape((-1))
        labels = train_data.label().astype(np.int32)

        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=CanonicalAtomFeaturizer()
        )

        graph_list = []
        for smile in smiles:
            graph_list.append(smiles_to_g(smile))

        valid_ids = []
        clean_graphs = []
        failed_mols = []
        clean_labels = []
        clean_smiles = []
        for i, g in enumerate(graph_list):
            if g is not None:
                valid_ids.append(i)
                clean_graphs.append(g)
                clean_labels.append(labels[i])
                clean_smiles.append(smiles[i])
            else:
                failed_mols.append((i, smiles[i]))

        # construct dataset
        if train_data:
            trainset = SmilesDataset(
                clean_smiles, clean_graphs, torch.Tensor(clean_labels))

        # construct dataloader
        batch_size = self.common_config.train_params.get("train_batch_size")
        if trainset:
            train_dataloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_molgraphs
            )
        return train_dataloader

    def _set_val_dataloader(self):
        val_data = self._read_data(self.common_config.input_valset)
        valset = None
        val_dataloader = None
        if val_data is None:
            return val_dataloader

        # construct smiles
        smiles = val_data.features(type="series").values.reshape((-1))
        labels = val_data.label().astype(np.int32)

        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=CanonicalAtomFeaturizer()
        )

        graph_list = []
        for smile in smiles:
            graph_list.append(smiles_to_g(smile))

        valid_ids = []
        clean_graphs = []
        failed_mols = []
        clean_labels = []
        clean_smiles = []
        for i, g in enumerate(graph_list):
            if g is not None:
                valid_ids.append(i)
                clean_graphs.append(g)
                clean_labels.append(labels[i])
                clean_smiles.append(smiles[i])
            else:
                failed_mols.append((i, smiles[i]))

        # construct dataset
        if val_data:
            valset = SmilesDataset(
                clean_smiles, clean_graphs, torch.Tensor(clean_labels))

        # construct dataloader
        batch_size = self.common_config.train_params.get("val_batch_size")
        if valset:
            val_dataloader = DataLoader(
                valset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_molgraphs
            )
        return val_dataloader

    def val_loop(self, dataset_type: str = "val", context: dict = {}):
        self.model.eval()
        val_loss = 0
        val_predicts = []
        labels = []

        lossfunc_name = list(self.lossfunc.keys())[0]
        lossfunc = list(self.lossfunc.values())[0]

        if dataset_type == "val":
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for batch, (smiles, bg, label, masks) in enumerate(dataloader):
            node_feats = bg.ndata.pop('h')
            logits = self.model(bg, node_feats)
            label = label.reshape((-1, 1))
            loss = lossfunc(logits, label)

            val_predicts.append(logits.detach().cpu().squeeze(-1).numpy())
            val_loss += loss.item()

            labels.append(label.cpu().squeeze(-1).numpy())

        val_loss /= len(dataloader)
        labels: np.ndarray = np.concatenate(labels, axis=0)
        val_predicts: np.ndarray = np.concatenate(val_predicts, axis=0)
        if len(val_predicts.shape) == 1:
            val_predicts = np.array(val_predicts > 0.5, dtype=np.int32)
        elif len(val_predicts.shape) == 2:
            val_predicts = val_predicts.argmax(axis=-1)

        metrics_output = CommonMetrics._calc_metrics(
            metrics=self.metrics,
            labels=labels,
            val_predicts=val_predicts,
            lossfunc_name=lossfunc_name,
            loss=val_loss,
            dataset_type=dataset_type
        )

        global_epoch = self.context["g_epoch"]
        if dataset_type == "val":
            local_epoch = None
        elif dataset_type == "train":
            local_epoch = self.context["l_epoch"]

        CommonMetrics.save_metric_csv(
            metrics_output=metrics_output, 
            output_config=self.common_config.output, 
            global_epoch=global_epoch, 
            local_epoch=local_epoch, 
            dataset_type=dataset_type,
        )

        early_stop_flag = self.context["early_stop_flag"]
        if (self.common_config.save_frequency > 0) & \
            (dataset_type == "val") & (self.earlystopping.patience > 0):
            early_stop_flag = self.earlystopping(metrics_output, global_epoch)
            if early_stop_flag:
                # find the saved epoch closest to the best epoch
                best_epoch = self.earlystopping.best_epoch
                closest_epoch = round(best_epoch / self.common_config.save_frequency) * \
                    self.common_config.save_frequency
                closest_epoch -= self.common_config.save_frequency \
                    if closest_epoch > global_epoch else 0
                self.context["early_stop_flag"] = True
                self.context["early_stop_epoch"] = closest_epoch
    