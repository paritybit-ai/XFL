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
from torch.utils.data import DataLoader, TensorDataset, Dataset

from algorithm.core.data_io import CsvReader, NpzReader
from common.utils.logger import logger
import torchvision.transforms as transforms
from PIL import Image

import dgl
from dgllife.model import GCNPredictor
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer


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


class Common():
    def _set_model(self) -> nn.Module:
        model_config = self.model_info.get("config")
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
        train_data = self._read_data(self.input_trainset)
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
        batch_size = self.train_params.get("batch_size", 64)
        if trainset:
            train_dataloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_molgraphs
            )
        return train_dataloader

    def _set_val_dataloader(self):
        val_data = self._read_data(self.input_valset)
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
        batch_size = self.train_params.get("batch_size", 64)
        if valset:
            val_dataloader = DataLoader(
                valset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_molgraphs
            )
        return val_dataloader

    def val_loop(self, dataset_type: str = "validation", context: dict = {}):
        self.model.eval()
        val_loss = 0
        val_predicts = []
        labels_list = []
        metric_output = {}

        loss_func_name = list(self.loss_func.keys())[0]
        loss_func = list(self.loss_func.values())[0]

        if dataset_type in ["validation", "val"]:
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for batch, (smiles, bg, labels, masks) in enumerate(dataloader):
            node_feats = bg.ndata.pop('h')
            logits = self.model(bg, node_feats)
            labels = labels.reshape((-1, 1))
            loss = loss_func(logits, labels)

            val_predicts.append(logits.detach().cpu().squeeze(-1).numpy())
            val_loss += loss.item()

            labels_list.append(labels.cpu().squeeze(-1).numpy())

        val_loss /= len(dataloader)
        metric_output[loss_func_name] = val_loss

        val_predicts = np.concatenate(val_predicts, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)
        if len(val_predicts.shape) == 1:
            val_predicts = np.array(val_predicts > 0.0, dtype=np.int32)
        elif len(val_predicts.shape) == 2:
            val_predicts = val_predicts.argmax(axis=-1)

        metrics_conf: dict = self.train_params["metric_config"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](
                labels_list, val_predicts, **metrics_conf[method])
        logger.info(f"Metrics on {dataset_type} set: {metric_output}")
