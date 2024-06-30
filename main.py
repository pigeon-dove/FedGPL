# %%
import os
import time
import torch
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
import itertools

from llm_tuning.dataset import Gsm8kDataset, MathDataset, SVAMPDataset, MultiArithDataset
from llm_tuning.train import LlmTrainer
from llm_tuning.train_fed import LlmFedTrainer
from llm_tuning.model import get_4bit_model, get_lora_model, get_tokenizer, get_ptuning_model, get_prompt_model
from llm_tuning.train_fed_split import LlmFedSplitTrainer
from llm_tuning.utils import set_seed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

seed = 3407

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", default=time.strftime("%y%m%d-%H%M", time.localtime(time.time())), type=str)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument("--device", default="cuda:1", type=str)

    parser.add_argument("--token", default="", type=str)

    parser.add_argument("--fed_alg", default="FedAdam", type=str, choices=["FedAdam", "FedAVG", "FedProx"])
    parser.add_argument("--peft", default="lora", type=str, choices=["lora", "p-tuning", "prompt-tuning"])
    parser.add_argument("--client_num", default=50, type=int)
    parser.add_argument("--client_num_per_step", default=4, type=int)
    parser.add_argument("--data_name", default="gsm8k", type=str,
                        choices=["gsm8k", "camel-ai/math", "ChilleD/SVAMP", "ChilleD/MultiArith"])
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf", type=str,
                        choices=["meta-llama/Llama-2-7b-chat-hf",
                                 "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                 "bigscience/bloom-3b"])

    parser.add_argument("--client_epoch", default=1, type=int)
    parser.add_argument("--client_batch_per_step", default=8, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accum_steps", default=4, type=int)

    parser.add_argument("--max_steps", default=6000, type=int)
    parser.add_argument("--val_steps", default=100, type=int)

    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--client_lr", default=1e-4, type=float)
    parser.add_argument("--train_mode", default="fedGradFocus", type=str, choices=["local", "fed", "fedGradFocus"])
    parser.add_argument("--max_layer_num", default=6, type=int)
    parser.add_argument("--min_layer_num", default=2, type=int)

    parser.add_argument("--grad_eval", default="lora_per_l1", type=str, choices=["l1", "l2", "lora_l1", "lora_l2", "lora_per_l1", "lora_per_l2"])
    return parser.parse_args()


config = parse_args()

# %%
token = config.token
model = get_4bit_model(config.model_name, token, config.device)

if config.peft == "lora":
    model = get_lora_model(model)
elif config.peft == "p-tuning":
    model = get_ptuning_model(model)
elif config.peft == "prompt-tuning":
    model = get_prompt_model(model)
model.print_trainable_parameters()

tokenizer = get_tokenizer(config.model_name, token)

set_seed(seed)

data_name = config.data_name
if data_name == "gsm8k":
    full_dataset = ConcatDataset([
        Gsm8kDataset(load_dataset(data_name, "main", split="train"), tokenizer, max_length=512, model_name=config.model_name),
        Gsm8kDataset(load_dataset(data_name, "main", split="test"), tokenizer, max_length=512, model_name=config.model_name)
    ])
elif data_name == "camel-ai/math":
    full_dataset = MathDataset(load_dataset(data_name, split="train"), tokenizer, max_length=512, model_name=config.model_name)
elif data_name == "ChilleD/SVAMP":
    full_dataset = ConcatDataset([
        SVAMPDataset(load_dataset(data_name, split="train"), tokenizer, max_length=300, model_name=config.model_name),
        SVAMPDataset(load_dataset(data_name, split="test"), tokenizer, max_length=300, model_name=config.model_name)
    ])
elif data_name == "ChilleD/MultiArith":
    full_dataset = ConcatDataset([
        MultiArithDataset(load_dataset(data_name, split="train"), tokenizer, max_length=256, model_name=config.model_name),
        MultiArithDataset(load_dataset(data_name, split="test"), tokenizer, max_length=256, model_name=config.model_name)
    ])
else:
    raise f"dataset {data_name} not found"

total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

# %%
trainer = None
if config.train_mode == "fed":
    trainer = LlmFedTrainer(model, tokenizer, train_ds, val_dl, config)
elif config.train_mode == "fedGradFocus":
    trainer = LlmFedSplitTrainer(model, tokenizer, train_ds, val_dl, config)
elif config.train_mode == "local":
    trainer = LlmTrainer(model, tokenizer, train_ds, val_dl, config)
trainer.train()
