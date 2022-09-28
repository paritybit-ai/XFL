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
import shutil
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from algorithm.core.data_io import CsvReader

import numpy as np
import pandas as pd
import pytest


from service.fed_config import FedConfig
from service.fed_job import FedJob
from service.fed_node import FedNode
from algorithm.core.horizontal.aggregation.aggregation_plain import AggregationPlainRoot, AggregationPlainLeaf
from common.communication.gRPC.python.channel import DualChannel
from common.communication.gRPC.python.commu import Commu
from algorithm.core.horizontal.template.torch.fedtype import _get_assist_trainer, _get_label_trainer
from algorithm.model.logistic_regression import LogisticRegression

MOV = b"@" # middle of value
EOV = b"&" # end of value

def prepare_data():
    case_df = pd.DataFrame({
        'x0': np.random.random(1000),
        'x1': [0] * 1000,
        'x2': 2 * np.random.random(1000) + 1.0,
        'x3': 3 * np.random.random(1000) - 1.0,
        'x4': np.random.random(1000)
    })
    case_df['y'] = np.where(case_df['x0'] + case_df['x2'] + case_df['x3'] > 2.5, 1, 0)
    case_df = case_df[['y', 'x0', 'x1', 'x2', 'x3', 'x4']]
    case_df.head(800).to_csv(
        "/opt/dataset/unit_test/train_data.csv", index=True
    )
    case_df.tail(200).to_csv(
        "/opt/dataset/unit_test/test_data.csv", index=True
    )

@pytest.fixture()
def get_assist_trainer_conf():
    with open("python/algorithm/config/horizontal_logistic_regression/assist_trainer.json") as f:
        conf = json.load(f)
        conf["input"]["valset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["valset"][0]["name"] = "test_data.csv"
        conf["output"]["model"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
    yield conf

@pytest.fixture()
def get_trainer_conf():
    with open("python/algorithm/config/horizontal_logistic_regression/trainer.json") as f:
        conf = json.load(f)
        conf["input"]["trainset"][0]["path"] = "/opt/dataset/unit_test"
        conf["input"]["trainset"][0]["name"] = "train_data.csv"
        conf["output"]["metrics"]["path"] = "/opt/checkpoints/unit_test"
        conf["output"]["evaluation"]["path"] = "/opt/checkpoints/unit_test"
    yield conf


@pytest.fixture(scope="module", autouse=True)
def env():
    if not os.path.exists("/opt/dataset/unit_test"):
        os.makedirs("/opt/dataset/unit_test")
    if not os.path.exists("/opt/checkpoints/unit_test"):
        os.makedirs("/opt/checkpoints/unit_test")
    prepare_data()
    yield
    if os.path.exists("/opt/dataset/unit_test"):
        shutil.rmtree("/opt/dataset/unit_test")
    if os.path.exists("/opt/checkpoints/unit_test"):
        shutil.rmtree("/opt/checkpoints/unit_test")


class TestAggregation:
    @pytest.mark.parametrize("aggregation_method", ["fedavg", "fedprox"])
    def test_trainer(self, get_trainer_conf, get_assist_trainer_conf, aggregation_method, mocker):
        fed_method = None
        fed_assist_method = None
        mocker.patch.object(Commu, "node_id", "assist_trainer")
        Commu.trainer_ids = ['node-1', 'node-2']
        Commu.scheduler_id = 'assist_trainer'
        conf = get_trainer_conf
        assist_conf = get_assist_trainer_conf
        conf["model_info"]["config"]["input_dim"] = 5
        assist_conf["model_info"]["config"]["input_dim"] = 5
        conf["train_info"]["params"]["aggregation_config"]["type"] = aggregation_method
        conf["train_info"]["params"]["aggregation_config"]["mu"] = 0.01
        assist_conf["train_info"]["params"]["aggregation_config"]["type"] = aggregation_method

        conf["train_info"]["params"]["aggregation_config"]["encryption"] = {"method": "plain"}
        assist_conf["train_info"]["params"]["aggregation_config"]["encryption"] = {"method": "plain"}
        sec_conf = conf["train_info"]["params"]["aggregation_config"]["encryption"]

        mocker.patch.object(FedConfig, "get_label_trainer", return_value=['node-1', 'node-2'])
        mocker.patch.object(FedConfig, "get_assist_trainer", return_value='assist_trainer')
        mocker.patch.object(FedConfig, "node_id", 'node-1')
        FedConfig.stage_config = conf
        assist_trainer = _get_assist_trainer()
        label_trainer = _get_label_trainer()
        fed_method = AggregationPlainLeaf(sec_conf)
        fed_assist_method = AggregationPlainRoot(sec_conf)

        def mock_recv(*args, **kwargs):
            return params_plain_recv

        def mock_agg(*args, **kwargs):
            return agg_otp

        def mock_set_model():
            model_config = conf["model_info"]["config"]
            model = LogisticRegression(input_dim=model_config["input_dim"], bias=model_config["bias"])
            return model

        def _read_data(input_dataset):
            conf = input_dataset[0]
            path = os.path.join(conf['path'], conf['name'])
            has_label = conf["has_label"]
            has_id = conf['has_id']
            return CsvReader(path, has_id, has_label)
        label_trainer._read_data = _read_data
        assist_trainer._read_data = _read_data

        def generate_dataset(trainer, config):
            trainer.train_conf = config
            trainer.identity = config.get("identity")
            trainer.fed_config = config.get("fed_info")
            trainer.model_info = config.get("model_info")
            trainer.inference = config.get("inference", False)
            trainer.train_info = config.get("train_info")
            trainer.extra_info = config.get("extra_info")
            trainer.computing_engine = config.get("computing_engine", "local")
            trainer.device = trainer.train_info.get("device", "cpu")
            trainer.train_params = trainer.train_info.get("params")
            trainer.interaction_params = trainer.train_info.get("interaction_params", {})
            trainer.input = config.get("input")
            for i in ["dataset", "trainset", "valset", "testset"]:
                for j in range(len(trainer.input.get(i, []))):
                    if "path" in trainer.input[i][j]:
                        trainer.input[i][j]["path"] = trainer.input[i][j]["path"] \
                            .replace("[JOB_ID]", str(FedJob.job_id)) \
                            .replace("[NODE_ID]", str(FedNode.node_id))
            trainer.input_trainset = trainer.input.get("trainset", [])
            trainer.input_valset = trainer.input.get("valset", [])
            trainer.input_testset = trainer.input.get("testset", [])
        generate_dataset(label_trainer, conf)
        generate_dataset(assist_trainer, assist_conf)

        def mock_label_train_dataloader():
            train_data = label_trainer._read_data(label_trainer.input_trainset)   
            if train_data:
                trainset = TensorDataset(torch.tensor(train_data.features(), dtype=torch.float32).to(label_trainer.device),
                torch.tensor(train_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(label_trainer.device))
            batch_size = label_trainer.train_params.get("batch_size", 64)
            if trainset:
                train_dataloader = DataLoader(trainset, batch_size, shuffle=True)
            return train_dataloader

        def mock_dataloader():
            pass

        def mock_assist_val_dataloader():
            val_data = assist_trainer._read_data(assist_trainer.input_valset)
            if val_data:
                valset = TensorDataset(torch.tensor(val_data.features(), dtype=torch.float32).to(assist_trainer.device),
                torch.tensor(val_data.label(), dtype=torch.float32).unsqueeze(dim=-1).to(assist_trainer.device))
            batch_size = assist_trainer.train_params.get("batch_size", 64)
            if valset:
                val_dataloader = DataLoader(valset, batch_size, shuffle=True)
            return val_dataloader

        mocker.patch.object(label_trainer, "_set_model", side_effect=mock_set_model)
        mocker.patch.object(assist_trainer, "_set_model", side_effect=mock_set_model)
        mocker.patch.object(label_trainer, "_set_train_dataloader", side_effect=mock_label_train_dataloader)
        mocker.patch.object(assist_trainer, "_set_train_dataloader", side_effect=mock_dataloader)
        mocker.patch.object(label_trainer, "_set_val_dataloader", side_effect=mock_dataloader)
        mocker.patch.object(assist_trainer, "_set_val_dataloader", side_effect=mock_assist_val_dataloader)

        lrt = label_trainer(conf)
        lrt_a = assist_trainer(assist_conf)

        params_plain_recv = pickle.dumps(lrt_a.model.state_dict()) + EOV
        params_send = fed_method._calc_upload_value(lrt.model.state_dict(), len(lrt.train_dataloader.dataset))
        params_collect = pickle.dumps(params_send)
        agg_otp = fed_assist_method._calc_aggregated_params(list(map(lambda x: pickle.loads(x), [params_collect,params_collect])))
        
        def mock_recv(*args, **kwargs):
            return params_plain_recv

        mocker.patch.object(DualChannel, "recv", side_effect=mock_recv)
        mocker.patch.object(DualChannel, "__init__", return_value=None)
        mocker.patch.object(DualChannel, "send", return_value=None)
        mocker.patch.object(AggregationPlainRoot, "aggregate", side_effect=mock_agg)
        
        def mock_train_loop():
            lrt.model.train()
            train_loss = 0
            loss_func = list(lrt.loss_func.values())[0]
            optimizer = list(lrt.optimizer.values())[0]
            lr_scheduler = list(lrt.lr_scheduler.values())[0] if lrt.lr_scheduler.values() else None
            for batch, (feature, label) in enumerate(lrt.train_dataloader):
                pred = lrt.model(feature)
                loss = loss_func(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(lrt.train_dataloader)
            if lr_scheduler:
                lr_scheduler.step()
            lrt.context["train_loss"] = train_loss

        def mock_train_loop_a():
            pass

        mocker.patch.object(label_trainer, "train_loop", side_effect=mock_train_loop)
        mocker.patch.object(assist_trainer, "train_loop", side_effect=mock_train_loop_a)

        lrt.fit()
        lrt_a.fit()