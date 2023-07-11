[ChatGLM](https://github.com/THUDM/ChatGLM-6B) is an open source, bilingual conversation language model based on the General Language Model (GLM) architecture with 6.2 billion parameters. ChatGLM-6B uses similar technology to ChatGPT and is optimised for Chinese Q&A and dialogue. With approximately 1T identifiers trained in both Chinese and English, supported by supervised fine-tuning, feedback self-help, and reinforcement learning with human feedback, ChatGLM-6B with 6.2 billion parameters is able to generate responses that are quite compatible with human preferences.

XFL supports ChatGLM lateral federation fine-tuning and provides two types of fine-tuning, lora, and ptuning-v2. This tutorial will describe how to configure the lateral ChatGLM federation operator and perform federation fine-tuning training.

# Dataset preparation

The horizontal ChatGLM operator supports datasets in json format and can be configured for datasets with the following configuration parameters:

```json
"dataset": {
    "max_src_length": 100,
    "max_dst_length": 100,
    "prompt_pattern":"{}：\n问：{}\n答：",
    "key_query": "input",
    "key_answer": "output"
}
```

An example of a dataset file format is as follows:

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

where `key_query` and `key_answer` denote the keywords for `query` and `answer` respectively, defaulting to `input` and `output`, and `prompt_pattern` denotes the padding format for `prompt`, which in the above example would be padded with: "{$instruction}：\n问：{$input}\n答：". `max_src_length` and `max_dst_length` indicate the maximum acceptable encoding length for `prompt` and `output` respectively after padding.

# Federal participant configuration

As with other XFL operators, configure the ip, port, etc. of the federated participants in `fed_conf.json`. An example is as follows:

```json
{
    "fed_info": {
       "scheduler": {
            "scheduler": "localhost:55005" //Stand-alone mode, in multi-node it needs to be set to individual node ip's
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

The above example is a single machine simulation of a federation configuration with two normal `trainers` and one aggregated party `assist_trainer`. When configuring across multiple parties, the `fed_info` should be consistent for each party and the user will need to replace the ip and port of each party and set the `redis_server` configuration as appropriate.

# Algorithm configuration

The operator configuration is set in the `trainer_config_{$NODE_ID}.json` file as follows:

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

Depending on the role of the participant, `identity` can be `assist_trainer` and `label_trainer`, where `assist_trainer` is the aggregator.

## input

Set the training set, pretrained model and adapter model in input. Where `pretrained_model` is a required field and `path` is the path to the pre-trained model folder. `adapter_model` is the pre-trained adapter model, which is used to continue training from the trained adapter model. `trainset` is the training set configuration and `path` can be set to the training set folder or file path.

Horizontal ChatGLM supports two training modes: 1) assist_trainer does not provide the training set and only aggregates; 2) assist_trainer provides the training set and also provides the aggregation function, where the model parameters on the assist_trainer node are aggregated directly without encryption.

## output

Set the path to the folder where the output model is stored in `path`.

## aggregation

Sets the aggregation frequency, which will aggregate and save models at intervals of `agg_steps` multiplied by the total local training steps for each participant.

```json
"aggregation": {
    "agg_steps": 0.2
}
```

## encryption

encryption supports `plain` unencrypted and `otp` encryption, where `otp` is configured as follows:

```json
"encryption": {
    "otp": {
        "key_bitlength": 64,  # 64 or 128
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

Two types of adapters are supported: `LORA`, `PREFIX_TUNING` (ptuning-v2). For `LORA` refer to the configuration in [Peft](https://github.com/huggingface/peft) and for `PREFIX_TUNING` refer to the configuration in [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) The parameters of ptuning are configured. Example configurations for both adapters are as follows:

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

This parameter is the one used when fine-tuning the Trainer, an example of which is as follows:

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

Please refer to [transformers](https://github.com/huggingface/transformers) for details of the parameters. Note that there is no need to set `save_steps` and `save_strategy`, `save_strategy` is built in as `steps` and `save_steps` will be calculated automatically based on `agg_steps`. No GPU is used when `no_cuda` is True, otherwise the GPU is used for fine tuning.

## dataset

See 'Data set preparation'

---

**Note**:

`label_trainer` only needs to be configured for some of the parameters in `trainer`, other parameters only need to be configured on the `assist_trainer` side.
The `trainer` parameters that can be configured by `label_trainer` are as follows:

```json
"trainer": {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_strategy": "steps",
    "torch_compile": false,
    "no_cuda": false
}
```

## Start fine-tuning

## Environment configuration

Start redis, using the command in a linux environment

```shell
redis-server &

cd PATH/TO/PROJECTHOME
source env.sh
```

## Stand-alone simulation

```
cd demo/horizontal/chatglm/3party
sh run.sh
```

## Multi-node runing

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

Fine-tuned models are stored in the `/opt/checkpoints/[JOB_ID]/` directory.

# Model Inference

Once fine-tuning is complete, the model can be inferred using the following code:

```python
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
