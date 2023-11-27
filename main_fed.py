# %%
import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from datasets import load_dataset

from llm_tuning.model import get_4bit_model, get_lora_model, get_tokenizer
from llm_tuning.train_fed import LlmFedTrainer
from llm_tuning.dataset import LlamaDataset
from llm_tuning.utils import set_seed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"
data_name = "gsm8k"
seed = 3407


@dataclass
class Config:
    exp_name: str = field(default=time.strftime("%y%m%d-%H%M", time.localtime(time.time())))

    client_num: int = field(default=50)
    client_num_per_step: int = field(default=4)

    client_epoch: int = field(default=1)
    client_batch_per_step: int = field(default=8)
    batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)

    lora_r: int = field(default=64)
    max_steps: int = field(default=3000)
    val_steps: int = field(default=50)
    max_length: int = field(default=512)
    lr: float = field(default=5e-4)


config = Config()

model = get_lora_model(get_4bit_model(model_name, token), lora_r=config.lora_r)
tokenizer = get_tokenizer(model_name, token)

# %%
set_seed(seed)

full_dataset = torch.utils.data.ConcatDataset([
    LlamaDataset(load_dataset(data_name, "main", split="train"), tokenizer, max_length=config.max_length),
    LlamaDataset(load_dataset(data_name, "main", split="test"), tokenizer, max_length=config.max_length)
])

total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

# %%
trainer = LlmFedTrainer(model, tokenizer, train_ds, val_dl, config)
trainer.train()
