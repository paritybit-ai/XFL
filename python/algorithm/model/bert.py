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



import tensorflow as tf
import tensorflow.keras as keras
from transformers import TFBertForSequenceClassification, BertConfig


class BertForSst2(keras.Model):
    def __init__(self, from_pretrained=True, num_labels=None, **kwargs):
        super().__init__()
        if from_pretrained:
            config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
            self.bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        else:
            config = BertConfig(num_labels=num_labels, **kwargs)
            self.bert = TFBertForSequenceClassification(config=config)
            self.bert(self.bert.dummy_inputs)
        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, input_ids, attention_mask, token_type_ids, labels):
        loss, logits = self.bert(input_ids = input_ids, attention_mask = attention_mask, 
                              token_type_ids=token_type_ids, labels = labels)[:2]
        prob = self.softmax(logits)
        return loss, logits, prob