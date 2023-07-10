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
import pytest
from pathlib import Path

from transformers import AutoTokenizer
    
from algorithm.core.data_io import QADataset


# @pytest.fixture(scope="module", autouse=True)
# def env(tmp_path):

def test_QADataset(tmp_path):
    data1 = {
        "id": "seed_task_0", 
        "name": "藏头诗", 
        "instruction": "根据提示词，写一首藏头诗。", 
        "instances": [
            {"input": "公携人地水风日长", "output": "公子申敬爱，携朋玩物华。人是平阳客，地即石崇家。水文生旧浦，风色满新花。日暮连归骑，长川照晚霞。"},
            {"input": "高下绮文蓂柳日连", "output": "高门引冠盖，下客抱支离。绮席珍羞满，文场翰藻摛。蓂华雕上月，柳色蔼春池。日斜归戚里，连骑勒金羁。"}
        ]
    }
    
    data2 = {
        "id": "seed_task_0", 
        "name": "藏头诗", 
        "instruction": "根据提示词，写一首藏头诗。", 
        "instances": [
            {"input": "砌津既方绮琼云丝", "output": "砌蓂收晦魄，津柳竞年华。既狎忘筌友，方淹投辖车。绮筵回舞雪，琼醑泛流霞。云低上天晚，丝雨带风斜。"},
            {"input": "春俯置开林竹兴山", "output": "春华归柳树，俯景落蓂枝。置驿铜街右，开筵玉浦陲。林烟含障密，竹雨带珠危。兴阑巾倒戴，山公下习池。"}
        ]
    }
    
    f1_path = tmp_path / "1.json"
    with open(f1_path, "w") as f1:
        json.dump(data1, f1)
        
    f2_path = tmp_path / "2.json"
    with open(f2_path, "w") as f2:
        json.dump(data2, f2)
        
    model_name_or_path = Path(__file__).parent / 'tokenizer'
    print(model_name_or_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)  # Callback
        
    dataset = QADataset(
        file_name_or_path=tmp_path,
        tokenizer=tokenizer,
        max_src_length=200,
        max_dst_length=500,
        prompt_pattern="{}：\n问：{}\n答：",
        key_query='input',
        key_answer='output'
    )
    
    assert len(dataset.data) == 4
    
    print(dataset[1])
