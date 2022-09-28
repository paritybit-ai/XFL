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
import pandas as pd
import numpy as np
from algorithm.model.bert import BertForSst2
from common.utils.logger import logger
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer

class Common():
    def _set_model(self) -> tf.keras.Model:
        model_config = self.model_info.get("config")
        model = BertForSst2(**model_config)
        return model
    
    def _read_data(self, input_dataset):
        if len(input_dataset) == 0:
            return None
        conf = input_dataset[0]
        path = os.path.join(conf['path'], conf['name'])
        raw_data = pd.read_csv(path, sep='\t')
        data = raw_data["sentence"].values, raw_data["label"].values
        return data

        
    def _set_train_dataloader(self):
        train_dataset = None
        train_dataloader = None
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_data = self._read_data(self.input_trainset)
        
        if train_data is not None:
            train_dataset = self._encode_examples(train_data, tokenizer)
            batch_size = self.train_params.get("batch_size", 64)
            train_dataloader = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        return train_dataloader
    
    def _set_val_dataloader(self):
        val_dataset = None
        val_dataloader = None
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        val_data = self._read_data(self.input_valset)
        if val_data is not None:
            val_dataset = self._encode_examples(val_data, tokenizer)
            batch_size = self.train_params.get("batch_size", 64)
            val_dataloader = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
        return val_dataloader
    
    def val_loop(self, dataset_type: str = "validation", context: dict = {}):
        val_loss = 0
        val_predicts = []
        labels = []
        metric_output = {}
        
        loss_func_name = list(self.loss_func.keys())[0]
        loss_func = list(self.loss_func.values())[0]
        
        if dataset_type in ["validation", "val"]:
            dataloader = self.val_dataloader
        elif dataset_type == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"dataset type {dataset_type} is not valid.")

        for idx, (input_ids, token_type_ids, attention_masks, label) in enumerate(tqdm(dataloader)):
            _,_,prob = self.model(input_ids, token_type_ids, attention_masks, label)
            loss = loss_func(label, prob)
            labels.append(label.numpy())
            val_predicts.append(tf.math.argmax(prob,-1).numpy())
            val_loss += float(loss)
            
        val_loss /= len(dataloader)
        metric_output[loss_func_name] = val_loss
        val_predicts = np.concatenate(val_predicts, axis=0)
        labels = np.concatenate(labels, axis=0)

        metrics_conf: dict = self.train_params["metric_config"]
        for method in self.metrics:
            metric_output[method] = self.metrics[method](labels, val_predicts, **metrics_conf[method])
        logger.info(f"Metrics on {dataset_type} set: {metric_output}")

    def _encode_examples(self, data, tokenizer, max_length=256):
        input_ids,token_type_ids,attention_masks,labels = [],[],[],[]
        for feature, label in zip(*data):
            bert_input = tokenizer.encode_plus(feature,
                            add_special_tokens = True, 
                            max_length = max_length, 
                            padding = 'max_length', 
                            return_token_type_ids = True,
                            return_attention_mask = True)

            input_ids.append(bert_input['input_ids'])
            token_type_ids.append(bert_input['token_type_ids'])
            attention_masks.append(bert_input['attention_mask'])
            labels.append([label])
        return tf.data.Dataset.from_tensor_slices((input_ids, token_type_ids, attention_masks, labels))