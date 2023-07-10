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


from typing import Tuple, Union

import numpy as np
import torch
from numpy.core.records import ndarray
from sklearn.metrics import auc, roc_curve
from torch.nn import Module

from common.utils.logger import logger


class MapeLoss(Module):
    def __init__(self):
        super(MapeLoss, self).__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
            Args:
                preds:
                labels:

            Returns:

            """
        mask = (labels != 0)
        distance = torch.abs(preds - labels) / torch.abs(labels)
        return torch.mean(distance[mask])


class BiClsAccuracy(Module):
    # torch mape loss function
    def __init__(self):
        super(BiClsAccuracy, self).__init__()

    def forward(self, confusion_matrix: np.array) -> torch.Tensor:
        """
            Binary Classification Accuracy
            Args:
                confusion_matrix:

            Returns:

            """
        tn, fp, fn, tp = confusion_matrix.ravel()
        return (tn + tp) / (tn + fp + fn + tp)


class BiClsPrecision(Module):
    def __init__(self):
        super(BiClsPrecision, self).__init__()

    def forward(self, confusion_matrix: np.array) -> torch.Tensor:
        """
            Binary Classification precision
            Args:
                confusion_matrix:

            Returns:

            """
        tn, fp, fn, tp = confusion_matrix.ravel()
        if fp + tp > 0:
            return tp / (fp + tp)
        else:
            return torch.Tensor([0.0])


class BiClsRecall(Module):
    def __init__(self):
        super(BiClsRecall, self).__init__()

    def forward(self, confusion_matrix: np.array) -> torch.Tensor:
        """
            Binary Classification recall
            Args:
                confusion_matrix:

            Returns:

            """
        tn, fp, fn, tp = confusion_matrix.ravel()
        if fn + tp > 0:
            return tp / (fn + tp)
        else:
            return torch.Tensor([0.0])


class BiClsF1(Module):
    def __init__(self):
        super(BiClsF1, self).__init__()

    def forward(self, confusion_matrix: np.array) -> torch.Tensor:
        """
            Binary Classification recall
            Args:
                confusion_matrix:

            Returns:

        """
        tn, fp, fn, tp = confusion_matrix.ravel()
        if fp + tp > 0 and fn + tp > 0:
            precision, recall = tp / (fp + tp), tp / (fn + tp)
            return 2 * precision * recall / (precision + recall)
        else:
            return torch.Tensor([0.0])


class BiClsAuc(Module):
    def __init__(self):
        super(BiClsAuc, self).__init__()

    def forward(self, tpr: np.array, fpr: np.array) -> float:
        """
        auc
        Args:
            tpr: TP / (TP + FN)
            fpr: FP / (FP + TN)

        Returns: auc_score

        """
        auc_score = auc(fpr, tpr)
        return auc_score


class BiClsKS(Module):
    def __init__(self):
        super(BiClsKS, self).__init__()

    def forward(self, tpr: np.array, fpr: np.array) -> float:
        """
        ks
        Args:
            tpr: TP / (TP + FN)
            fpr: FP / (FP + TN)

        Returns: ks

        """
        ks = max(np.max(tpr - fpr), 0)
        return ks


class aucScore(Module):
    def __init__(self):
        super(aucScore, self).__init__()

    def forward(self, pred: np.array, label: np.array) -> Tuple[float, Union[ndarray, int, float, complex]]:
        """
        auc
        Args:
            pred:
            label:

        Returns: auc_score, ks

        """
        fpr, tpr, _ = roc_curve(label, pred)
        auc_score = auc(fpr, tpr)
        ks = max(np.max(tpr - fpr), 0)
        return auc_score, ks


class earlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, key: str, patience: int = 10, delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.key = key
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metric) -> Tuple[bool, bool]:
        if self.key not in metric:
            raise KeyError("Key {} cannot found in metrics.".format(self.key))
        save_flag, val_score = False, metric[self.key]
        if self.best_score is None:
            self.best_score, save_flag = val_score, True
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}. Epoch score {val_score}, '
                f'best score {self.best_score}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score, save_flag = val_score, True
            self.counter = 0
        return self.early_stop, save_flag


class _earlyStopping:
    """Early stops the training if validation metric doesn't increase or decrease after a given patience."""

    def __init__(self, key: str, patience: int = 10, delta: float = 0, maxmize: bool = True):
        """
        Args:
            key (str): The key of metric to monitor.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            maxmize (bool): If True, we try to maxmize the metric. Otherwise, we try to minimize the metric.
        """
        self.patience = patience
        self.key = key
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.maxmize = 1 if maxmize else -1
        self.delta = delta * maxmize

    def __call__(self, metric: dict, epoch: int) -> bool:
        '''
        Args:
            metric (dict): The metric dict.
            epoch (int): The current epoch.
        '''
        if self.key not in metric:
            raise KeyError("Key {} cannot found in metrics.".format(self.key))
        val_score = metric[self.key]
        
        if self.best_score is None:
            # update best score and best epoch
            self.best_score = val_score
            self.best_epoch = epoch

        elif (val_score * self.maxmize) < ((self.best_score + self.delta) * self.maxmize):
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}. '
                f'Epoch {epoch} score {val_score}, '
                f'best epoch {self.best_epoch} best score {self.best_score}.')
            if (val_score * self.maxmize) < (self.best_score * self.maxmize):
                # update best score and best epoch
                self.best_score = val_score
                self.best_epoch = epoch
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


class earlyStoppingH(_earlyStopping):
    """Early stops the training if validation metric doesn't increase after a given patience."""

    def __init__(self, key: str, patience: int = 10, delta: float = 0):
        """
        Args:
            key (str): The key of metric to monitor.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        maxmize = None
        if key in ["acc", "precision", "recall", "f1_score", "auc", "ks"]:
            maxmize = True
        elif key in ["mae", "mse", "mape", "rmse"]:
            maxmize = False
        else:
            raise ValueError("Key {} cannot be monitored.".format(key))
        super().__init__(key, patience, delta, maxmize=maxmize)