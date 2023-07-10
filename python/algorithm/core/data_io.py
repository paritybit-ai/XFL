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
import json
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class CsvReader(object):
    def __init__(self,
                 path: str,
                 has_id: bool = True,
                 has_label: bool = True):
        index_col = 0 if has_id else False
        self.table: pd.DataFrame = pd.read_csv(path, index_col=index_col)
        self.ids = self.table.index.to_numpy()
        self.table.reset_index(drop=True, inplace=True)
        self.has_id = has_id
        self.has_label = has_label
        self.label_col = 0 if has_label else -1

    def features(self, type: str = "numpy.ndarray"):
        if type == "numpy.ndarray":
            return self.table.iloc[:, self.label_col + 1:].to_numpy().astype(np.float32)
        else:  # pandas.dataframe
            return self.table.iloc[:, self.label_col + 1:]

    def label(self, type: str = "numpy.ndarray"):
        if self.label_col == 0:
            if type == "numpy.ndarray":
                return self.table.iloc[:, 0].to_numpy().astype(np.float32)
            else:
                return self.table.iloc[:, 0]
        else:
            return None

    def col_names(self):
        return self.table.columns.tolist()

    def feature_names(self):
        index = 0
        if self.has_label:
            index += 1
        return self.table.columns.tolist()[index:]

    def label_name(self):
        if self.label_col == 0:
            return self.table.columns.tolist()[0]
        else:
            return None


class NpzReader(object):
    def __init__(self,
                 path: str):
        self.data = np.load(path, allow_pickle=True)

    def features(self):
        return self.data["data"].astype(float)

    def label(self):
        return self.data["labels"].astype(float)

    
class NdarrayIterator():
    def __init__(self, data: np.ndarray, batch_size: int):
        self.data = data
        self.bs = batch_size
        self.index = 0
        
    def __len__(self):
        return len(self.data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            data = self.data[self.index: self.index + self.bs]
            self.index += self.bs
            return data
        else:
            self.index = 0
            raise StopIteration

        
class QADataset(Dataset):
    def __init__(self,
                 file_name_or_path,
                 tokenizer,
                 max_src_length=200,
                 max_dst_length=500,
                 prompt_pattern="{}：\n问：{}\n答：",
                 key_query='input',
                 key_answer='output'):
        super().__init__()
        
        if os.path.isdir(file_name_or_path):
            data = []
            for file_name in os.listdir(file_name_or_path):
                with open(os.path.join(file_name_or_path, file_name), 'r') as fp:
                    content = json.load(fp)
                    instruction = content["instruction"]
                    instances = content["instances"]
                    for item in instances:
                        data.append(
                            {
                                "Q": prompt_pattern.format(instruction, item[key_query]),
                                "A": item[key_answer]
                            }
                        )
        elif os.path.isfile(file_name_or_path):
            data = []
            with open(file_name_or_path, 'r') as fp:
                content = json.load(fp)
                instruction = content["instruction"]
                instances = content["instances"]
                for item in instances:
                    data.append(
                        {
                            "Q": prompt_pattern.format(instruction, item[key_query]),
                            "A": item[key_answer]
                        }
                    )
        else:
            raise ValueError(f"Dataset path {file_name_or_path} is not a dir or a file name.")
        
        self.data = data
        self.tokenizer = tokenizer
        # self.prefix = prefix
        self.max_src_length = max_src_length
        self.max_dst_length = max_dst_length
        self.key_query = key_query
        self.key_answer = key_answer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        query, answer = self.data[index]["Q"], self.data[index]["A"]
        src_ids = self.tokenizer.encode(text=query, max_length=self.max_src_length, truncation=True)
        dst_ids = self.tokenizer.encode(text=answer, max_length=self.max_dst_length, truncation=True, add_special_tokens=False)
        input_ids = src_ids + dst_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(src_ids) + dst_ids + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels}
    
    
# class QADataset(Dataset):
#     def __init__(self,
#                  file_name_or_path,
#                  tokenizer,
#                  max_src_length=200,
#                  max_dst_length=500,
#                  ignore_pad_token_for_loss=True,
#                  prompt_pattern="{}：\n问：{}\n答：",
#                  key_query='input',
#                  key_answer='output'):
#         super().__init__()
        
#         if os.path.isdir(file_name_or_path):
#             data = []
#             for file_name in os.listdir(file_name_or_path):
#                 with open(os.path.join(file_name_or_path, file_name), 'r') as fp:
#                     content = json.load(fp)
#                     instruction = content["instruction"]
#                     instances = content["instances"]
#                     for item in instances:
#                         data.append(
#                             {
#                                 "Q": prompt_pattern.format(instruction, item[key_query]),
#                                 "A": item[key_answer]
#                             }
#                         )
#         elif os.path.isfile(file_name_or_path):
#             data = []
#             with open(file_name_or_path, 'r') as fp:
#                 content = json.load(fp)
#                 instruction = content["instruction"]
#                 instances = content["instances"]
#                 for item in instances:
#                     data.append(
#                         {
#                             "Q": prompt_pattern.format(instruction, item[key_query]),
#                             "A": item[key_answer]
#                         }
#                     )
#         else:
#             raise ValueError(f"Dataset path {file_name_or_path} is not a dir or a file name.")
        
#         self.data = data
#         self.tokenizer = tokenizer
#         # self.prefix = prefix
#         self.max_src_length = max_src_length
#         self.max_dst_length = max_dst_length
#         self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
#         self.key_query = key_query
#         self.key_answer = key_answer
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         query, answer = self.data[index]["Q"], self.data[index]["A"]
        
#         # prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
#         # target_ids = tokenizer.encode(
#         #     target,
#         #     max_length=max_seq_length,
#         #     truncation=True,
#         #     add_special_tokens=False)
#         # input_ids = prompt_ids + target_ids + [config.eos_token_id]
        
#         src_ids = self.tokenizer.encode(text=query, add_special_tokens=False)
#         dst_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
        
#         if len(src_ids) > self.max_src_length - 1:
#             src_ids = src_ids[: self.max_src_length - 1]

#         if len(dst_ids) > self.max_dst_length - 2:
#             dst_ids = dst_ids[: self.max_dst_length - 2]
        
#         input_ids = self.tokenizer.build_inputs_with_special_tokens(src_ids, dst_ids)
#         context_length = input_ids.index(self.tokenizer.bos_token_id)
#         mask_position = context_length - 1
#         labels = [-100] * context_length + input_ids[mask_position+1:]
        
#         # from original project code, is it necessary?
#         max_seq_length = self.max_src_length + self.max_dst_length
#         pad_len = max_seq_length - len(input_ids)
#         input_ids += [self.tokenizer.pad_token_id] * pad_len
#         labels += [self.tokenizer.pad_token_id] * pad_len
#         if self.ignore_pad_token_for_loss:
#             labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        
#         out = {
#             "input_ids": input_ids,
#             "labels": labels
#         }
#         return out
    
    
# class QADataset(Dataset):
#     """
#     [
#         {
#             "Q": "",
#             "A": ""
#         }
#     ]


#     """
#     def __init__(self,
#                  file_name_or_path,
#                  tokenizer,
#                  max_src_length=200,
#                  max_dst_length=500,
#                  ignore_pad_token_for_loss=True,
#                  key_query='input',
#                  key_answer='output'):
#         super().__init__()
        
#         if os.path.isdir(file_name_or_path):
#             data = []
#             for file_name in os.listdir(file_name_or_path):
#                 with open(os.path.join(file_name_or_path, file_name), 'r') as fp:
#                     content = json.load(fp)
#                     instruction = content["instruction"]
#                     instances = content["instances"]
#                     for item in instances:
#                         data.append(
#                             {
#                                 key_query: "{}：\n问：{}\n答：".format(instruction, item[key_query]),
#                                 key_answer: item[key_answer]
#                             }
#                         )
#         elif os.path.isfile(file_name_or_path):
#             data = []
#             with open(file_name_or_path, 'r') as fp:
#                 content = json.load(fp)
#                 instruction = content["instruction"]
#                 instances = content["instances"]
#                 for item in instances:
#                     data.append(
#                         {
#                             key_query: "{}：\n问：{}\n答：".format(instruction, item["input"]),
#                             key_answer: item["output"]
#                         }
#                     )
#         else:
#             raise ValueError(f"Dataset path {file_name_or_path} is not a dir or a file name.")
        
#         self.data = data
#         self.tokenizer = tokenizer
#         # self.prefix = prefix
#         self.max_src_length = max_src_length
#         self.max_dst_length = max_dst_length
#         self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
#         self.key_query = key_query
#         self.key_answer = key_answer
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         query, answer = self.data[index][self.key_query], self.data[index][self.key_answer]

#         # if self.prefix:
#         #     query = self.prefix + query
        
#         src_ids = self.tokenizer.encode(text=query, add_special_tokens=False)
#         dst_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
        
#         if len(src_ids) > self.max_src_length - 1:
#             src_ids = src_ids[: self.max_src_length - 1]

#         if len(dst_ids) > self.max_dst_length - 2:
#             dst_ids = dst_ids[: self.max_dst_length - 2]
        
#         input_ids = self.tokenizer.build_inputs_with_special_tokens(src_ids, dst_ids)
#         context_length = input_ids.index(self.tokenizer.bos_token_id)
#         mask_position = context_length - 1
#         labels = [-100] * context_length + input_ids[mask_position+1:]
        
#         # from original project code, is it necessary?
#         max_seq_length = self.max_src_length + self.max_dst_length
#         pad_len = max_seq_length - len(input_ids)
#         input_ids += [self.tokenizer.pad_token_id] * pad_len
#         labels += [self.tokenizer.pad_token_id] * pad_len
#         if self.ignore_pad_token_for_loss:
#             labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        
#         out = {
#             "input_ids": input_ids,
#             "labels": labels
#         }
#         return out

    
# def collate_fn_for_qa(batch):
#     input_ids = []
#     # attention_mask = []
#     labels = []
#     # position_ids = []
    
#     for obj in batch:
#         input_ids.append(obj['input_ids'])
#         labels.append(obj['labels'])
#         # attention_mask.append(obj['attention_mask'])
#         # position_ids.append(obj['position_ids'])
        
#     return {
#         'input_ids': torch.stack(input_ids),
#         'attention_mask': torch.stack(attention_mask), 
#         'labels': torch.stack(labels),
#         'position_ids':torch.stack(position_ids)
#     }