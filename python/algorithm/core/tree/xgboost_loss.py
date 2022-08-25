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


from typing import Optional

import numpy as np
import torch

from common.utils.constants import BCEWithLogitsLoss, MSELoss
from ..activation import sigmoid


def get_xgb_loss_inst(name: str, params: Optional[dict] = None):
    name = name.lower()
    if name == BCEWithLogitsLoss.lower():
        return XGBBCEWithLogitsLoss(params)
    elif name == MSELoss.lower():
        return XGBLoss(params)
    else:
        raise NotImplementedError(f"Loss {name} not implemented.")


class XGBLoss(object):
    def __init__(self, params: Optional[dict] = None):
        self.name = 'Loss'
        self.params = params

    def cal_grad(y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = True):
        raise NotImplementedError("Method cal_grad not implemented.")

    def cal_hess(y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = True):
        raise NotImplementedError("Method cal_hess not implemented.")
    
    # def predict(raw_value: np.ndarray):
    #     raise NotImplemented("Method predict not implemented.")
    
    def cal_loss(y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = False):
        raise NotImplementedError("Method cal_loss not implemented.")


class XGBBCEWithLogitsLoss(XGBLoss):
    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.name = BCEWithLogitsLoss
    
    def cal_grad(self, y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = True):
        if not after_prediction:
            y_pred = sigmoid(y_pred)
        return y_pred - y
    
    def cal_hess(self, y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = True):
        if not after_prediction:
            y_pred = sigmoid(y_pred)
        return y_pred * (1 - y_pred)
    
    def predict(self, raw_value: np.ndarray):
        return sigmoid(raw_value)
    
    def cal_loss(self, y: np.ndarray, y_pred: np.ndarray, after_prediction: bool = False):
        if not after_prediction:
            loss_func = torch.nn.BCEWithLogitsLoss()
            loss = loss_func(torch.tensor(y_pred), torch.tensor(y)).item()
        else:
            loss_func = torch.nn.BCELoss()
            loss = loss_func(torch.tensor(y_pred), torch.tensor(y)).item()
        return loss
        
        # if not after_prediction:
        #     y_pred = sigmoid(y_pred)
        # _loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        # loss = np.average(_loss)
        # return loss
    

class XGBMSELoss(XGBLoss):
    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.name = MSELoss

    def cal_grad(self, y: np.ndarray, y_pred: np.ndarray):
        return -2 * (y - y_pred)

    def cal_hess(self, y: np.ndarray, y_pred: np.ndarray):
        return 2
    
    def predict(self, raw_value: np.ndarray):
        return raw_value

    def cal_loss(self, y: np.ndarray, y_pred: np.ndarray):
        loss_func = torch.nn.MSELoss()
        loss = loss_func(torch.tensor(y_pred), torch.tensor(y)).item()
        # _loss = np.square(y - y_pred)
        # loss = np.average(_loss)
        return loss
