[ChatGLM](https://github.com/THUDM/ChatGLM-6B) 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答.

XFL支持ChatGLM横向联邦微调，提供了lora、ptuning-v2两种微调方式。本教程将介绍如何配置横向ChatGLM联邦算子并进行联邦微调训练。

# 数据集准备

横向ChatGLM算子支持json格式的数据集，并可对数据集进行配置，配置参数如下：
```json
"dataset": {
    "max_src_length": 100,
    "max_dst_length": 100,
    "prompt_pattern":"{}：\n问：{}\n答：",
    "key_query": "input",
    "key_answer": "output"
}
```

数据集文件格式示例如下：

```json
{
    "instruction": "根据提示词，写一首藏头诗。", 
    "instances": [
        {
            "input": "公携人地水风日长", 
            "output": "公子申敬爱，携朋玩物华。人是平阳客，地即石崇家。水文生旧浦，风色满新花。日暮连归骑，长川照晚霞。"
        }, 
        {
            "input": "忆平冠冲落朝谁珠", 
            "output": "忆昔江南年盛时，平生怨在长洲曲。冠盖星繁江水上，冲风摽落洞庭渌。落花两袖红纷纷，朝霞高阁洗晴云。谁言此处婵娟子，珠玉为心以奉君。"
        }
    ]
}

```
其中，`key_query`和`key_answer`分别表示`query`和`answer`的关键字，默认为`input`和`output`，`prompt_pattern`表示`prompt`的填充格式，在上例中，其内容将被填充为:"{$instruction}：\n问：{$input}\n答：". `max_src_length`和`max_dst_length`分别表示填充后`prompt`和`output`的最大可接受编码长度。

# 联邦参与方配置
与其他XFL算子相同，在`fed_conf.json`中配置联邦参与方的ip、port等信息。示例如下：
```json
{
    "fed_info": {
       "scheduler": {
            "scheduler": "localhost:55005" //单机模式，多机模式需要设置成各个节点的ip
        },
        "trainer": {
            "node-1": "localhost:56001",
            "node-2": "localhost:56002"
        },
        "assist_trainer": {
            "assist_trainer": "localhost:55004"
        }
    },
    "redis_server": "localhost:6379",
    "grpc": {
        "use_tls": false
    } 
}
```
上例为单机模拟联邦的配置，该配置中有两个普通`trainer`和一个聚合方`assist_trainer`。当在多方进行配置时，每方的`fed_info`应保持一致，用户需要根据实际情况替换各方的ip和port和设置`redis_server`的配置。

# 算子配置
算子配置在`trainer_config_{$NODE_ID}.json`文件中设置：
```json
[
    {
        "identity": ...,
        "model_info": {
            "name": "horizontal_chatglm"
        },
        "input": {
            "trainset": [
                {
                    "type": "QA",
                    "path": ...
                }
            ],
            "pretrained_model": {
                "path": ...
            },
            "adapter_model": {
                "path": ...
            }
        },
        "output": {
            "path":  ...
        },
        "train_info": {
            "train_params": {
                "aggregation": {
                    ...
                },
                "encryption": {
                    ...
                },
                "peft": {
                    ...
                },
                "trainer": {   
                    ...
                },
                "dataset": {
                    ...
                }
            }
        }
    }
]
```
根据参与方的角色，`identity`可以为`assist_trainer`和`label_trainer`，其中`assist_trainer`为聚合方。

## input
在input中设置训练集，预训练模型和adapter模型。其中`pretrained_model`为必填项，`path`为预训练模型文件夹路径。`adapter_model`为预训练好的adapter模型，用于从已训练的adapter模型继续训练。`trainset`为训练集配置，`path`可以设置为训练集文件夹或者文件路径。

横向ChatGLM支持两种训练模式：1) assist_trainer不提供训练集，仅进行聚合；2) assist_trainer提供训练集，同时提供聚合功能，此时assist_trainer节点上的模型参数不加密直接进行聚合。

## output
在`path`中设置输出模型保存的文件夹路径。

## aggregation
设置聚合频率，将按照每个参与方本地总训练步长乘以`agg_steps`的间隔来进行聚合和保存模型。
```json
"aggregation": {
    "agg_steps": 0.2
}
```

## encryption
encryption支持`plain`无加密和`otp`加密方式，其中`otp`的配置如下：
```json
"encryption": {
    "otp": {
        "key_bitlength": 64,  # 64或者128
        "data_type": "torch.Tensor",
        "key_exchange": {
            "key_bitlength": 3072,
            "optimized": true
        },
        "csprng": {
            "name": "hmac_drbg",
            "method": "sha512"
        }
    }
}
```

## peft
配置adapter参数，支持两种adapter：`LORA`, `PREFIX_TUNING`(ptuning-v2)。其中`LORA`的配置参考[Peft](https://github.com/huggingface/peft)的配置，`PREFIX_TUNING`的配置参考[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)中ptuning的参数配置。两种adapter的配置示例如下：
```json
# LORA
"peft": {
    "LORA": {
        "task_type": "CAUSAL_LM",
        "r": 8,
        "target_modules": ["query_key_value"],
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "fan_in_fan_out": false,
        "bias": "none",
        "modules_to_save": null
    }
}

# PREFIX_TUNING
"peft": {
    "PREFIX_TUNING": {
        "task_type": "CAUSAL_LM",
        "pre_seq_len": 20,
        "prefix_projection": false
    }
}

```

## trainer
该参数为Trainer微调时使用的参数，示例如下：
```json
"trainer": {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "num_train_epochs": 1,
    "save_strategy": "steps",
    "torch_compile": false,
    "no_cuda": false,
    "seed": 42
}
```
具体的参数说明请参考[transformers](https://github.com/huggingface/transformers). 注意无需设置`save_steps`和`save_strategy`, `save_strategy`内置为"steps"，`save_steps`将根据`agg_steps`自动计算。当`no_cuda`为True时不使用GPU，否则使用GPU进行微调。


## dataset
见`数据集准备`

---
**注意**:

`label_trainer`只需配置`trainer`中的部分参数，其他参数只需要`assist_trainer`一方配置即可。
`label_trainer`可配置的`trainer`参数如下：
```json
"trainer": {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_strategy": "steps",
    "torch_compile": false,
    "no_cuda": false
}
```

# 开始微调
## 环境配置
启动redis, linux环境中使用命令
```shell
redis-server &

cd PATH/TO/PROJECTHOME
source env.sh
```

## 单机模拟
```
cd demo/horizontal/chatglm/3party
sh run.sh
```

## 多机运行
```shell
# on node-1
python python/xfl.py -t node-1 --config_path demo/horizontal/chatglm/3party/config

# on node-2
python python/xfl.py -t node-1 --config_path demo/horizontal/chatglm/3party/config

# on assist-trainer node
python python/xfl.py -a --config_path demo/horizontal/chatglm/3party/config

# on arbitrary node
python python/xfl.py -s --config_path demo/horizontal/chatglm/3party/config

python python/xfl.py -c start --config_path demo/horizontal/chatglm/3party/config
```
微调模型保存在`/opt/checkpoints/[JOB_ID]/`目录下。

# 模型推断
微调完成后，可使用以下代码进行模型推断
``` python
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch, os
from peft import PeftModel
torch.cuda.empty_cache()

MODE="lora"
mpath = "/PATH/TO/ChatGLM-6b_model"
ptuning_path = "/PATH/TO/Ptuning-v2_model"
lora_path = "/PATH/TO/Lora_model"

# Replace it by your prompt
prompt = "根据提示词，写一首藏头诗。\n问：东南西北\n答：" 

if MODE=="original":
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModel.from_pretrained(mpath, trust_remote_code=True).half().cuda().eval()
elif MODE=="ptuning":
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    config = AutoConfig.from_pretrained(mpath, trust_remote_code=True, pre_seq_len=20) # pre_seq_len used in training
    model = AutoModel.from_pretrained(mpath, config=config, trust_remote_code=True)

    pd = torch.load(os.path.join(ptuning_path, "pytorch_model.bin"))
    new_dict = {}
    for k, v in pd.items():
        if k.startswith("transformer.prefix_encoder."):
            new_dict[k[len("transformer.prefix_encoder."):]] = v

    model = model.half()
    model.transformer.prefix_encoder.load_state_dict(new_dict)
    model = model.half().cuda().eval()
elif MODE=="lora":
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModel.from_pretrained(mpath, trust_remote_code=True)
    
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.half().cuda().eval()

response, history = model.chat(tokenizer, prompt, history=[])
print(response)

```


