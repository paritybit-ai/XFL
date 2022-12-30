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
from pathlib import Path

import pandas as pd

from common.evaluation.metrics import BiClsMetric, DecisionTable, RegressionMetric
from common.utils.config_parser import TrainConfigParser
from common.utils.logger import logger


class VerticalModelBase(TrainConfigParser):
    def __init__(self, train_conf: dict, label: bool = False):
        super().__init__(train_conf)
        self._parse_config()
        self.train_conf = train_conf
        self.label = label

    def _parse_config(self) -> None:
        # output_path
        # self.save_dir = Path(self.output.get("model", {}).get("path", ""))
        self.save_dir = Path(self.output.get("path", ""))
        self.metric_dir = self.save_dir
        # params
        self.lossfunc_conifg = self.train_params.get("lossfunc", {})
        self.metric_config = self.train_params.get("metric", {})
        # interaction_params
        self.echo_training_metrics = self.interaction_params.get("echo_training_metrics", False)
        self.write_training_prediction = self.interaction_params.get("write_training_prediction", False)
        self.write_validation_prediction = self.interaction_params.get("write_validation_prediction", False)
        # if self.output.get("metrics"):
        # 	self.metric_path = Path(self.output["metrics"].get("path"))
        # else:
        # 	self.metric_path = self.save_dir
        # # params
        # self.lossfunc_conifg = self.train_params.get("lossfunc_config")
        # self.metric_config = self.train_params.get("metric_config")
        # # interaction_params
        # self.echo_training_metrics = self.interaction_params.get("echo_training_metrics")
        # self.write_training_prediction = self.interaction_params.get("write_training_prediction")
        # self.write_validation_prediction = self.interaction_params.get("write_validation_prediction")

    def _calc_metrics(self, y, p, epoch, stage="train", loss={}):
        if stage == "train" and not self.echo_training_metrics:
            return
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)
        # output_file = os.path.join(
        #     self.metric_path, "{}_metrics.csv".format(stage))

        output_file = os.path.join(self.metric_dir, self.output.get("metric_" + stage)["name"])
        if self.model_info["name"] != "vertical_linear_regression":
            if loss:
                evaluate = BiClsMetric(epoch, output_file, self.metric_config)
            else:
                evaluate = BiClsMetric(epoch, output_file, self.metric_config, self.lossfunc_conifg)
        else:
            evaluate = RegressionMetric(epoch, output_file, self.metric_config)
        evaluate.calc_metrics(y, p)
        for key, value in loss.items():
            evaluate.metrics[key] = value
        if self.model_info["name"] != "vertical_linear_regression":
            evaluate.save()
        else:
            evaluate.save(evaluate.metrics)
        if "decision_table" in self.metric_config:
            dt = DecisionTable(self.metric_config["decision_table"])
            dt.fit(y, p)
            dt.save(os.path.join(self.metric_dir, self.output.get("decision_table_" + stage)["name"]))
            # dt.save(os.path.join(self.metric_path,
            #         "{}_decision_table.csv".format(stage)))
        logger.info("{} {}".format(stage, evaluate))
        return evaluate.metrics

    def _write_prediction(self, y, p, idx=None, epoch=None, final=False, stage="train"):
        if stage == "train" and not self.write_training_prediction:
            return
        elif stage == "val" and not self.write_validation_prediction:
            return
        elif stage not in ("train", "val"):
            raise ValueError("stage must be 'train' or 'val'.")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if final:
            # file_name = os.path.join(self.save_dir, "predicted_probabilities_{}.csv".format(stage))
            file_name = os.path.join(
                self.save_dir, self.output.get("prediction_" + stage)["name"])
        else:
            # file_name = os.path.join(self.save_dir, "predicted_probabilities_{}.epoch_{}".format(stage, epoch))
            file_name = self.output.get("prediction_" + stage)["name"]
            file_name_list = file_name.split(".")
            # file_name_list.insert(-1, '_epoch_'+str(epoch))
            file_name_list[-2] += '_epoch_' + str(epoch)
            file_name = '.'.join(file_name_list)
            file_name = os.path.join(self.save_dir, file_name)
        if idx is None:
            df = pd.DataFrame({"pred": p, "label": y})
            df = df.reset_index().rename(columns={"index": "id"})
            df.to_csv(file_name, header=True, index=False, float_format='%.6g')
        else:
            df = pd.DataFrame({'id': idx, "pred": p, "label": y})
            df.to_csv(file_name, header=True, index=False, float_format='%.6g')
