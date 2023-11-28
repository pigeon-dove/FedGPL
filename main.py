# %%
import argparse
import os
import time
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from llm_tuning.model import get_4bit_model, get_lora_model, get_tokenizer
from llm_tuning.train import LlmTrainer
from llm_tuning.dataset import LlamaDataset, inf_dl_gen
from llm_tuning.train_fed import LlmFedTrainer
from llm_tuning.utils import set_seed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"
data_name = "gsm8k"
seed = 1247


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default=time.strftime("%y%m%d-%H%M", time.localtime(time.time())), type=str)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--client_num", default=50, type=int)
    parser.add_argument("--client_num_per_step", default=4, type=int)

    parser.add_argument("--client_epoch", default=1, type=int)
    parser.add_argument("--client_batch_per_step", default=8, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accum_steps", default=2, type=int)

    parser.add_argument("--max_steps", default=6000, type=int)
    parser.add_argument("--val_steps", default=100, type=int)

    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--client_lr", default=1e-4, type=float)
    parser.add_argument("--train_mode", default="local", type=str, choices=["local", "fed"])
    return parser.parse_args()


config = parse_args()

# %%
model = get_lora_model(get_4bit_model(model_name, token, config.device), lora_r=64)
tokenizer = get_tokenizer(model_name, token)

set_seed(seed)
full_dataset = torch.utils.data.ConcatDataset([
    LlamaDataset(load_dataset(data_name, "main", split="train"), tokenizer, max_length=512),
    LlamaDataset(load_dataset(data_name, "main", split="test"), tokenizer, max_length=512)
])
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
elif config.train_mode == "local":
    trainer = LlmTrainer(model, tokenizer, train_ds, val_dl, config)
trainer.train()
