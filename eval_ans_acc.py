# %%
import math
import os
import re
import time

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM

from llm_tuning.dataset import LlamaDataset
from llm_tuning.utils import set_seed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"
data_name = "gsm8k"

exp_name = "gsm8k-origin"
# weights_file = "weights-250"
seed = 3407
batch_size = 2
instruction = "Please solve the following math problem and provide the answer in the format '### <number>' at the end."

# %%
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    token=token,
)

tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
tokenizer.cls_token = tokenizer.eos_token
tokenizer.mask_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, f"./result/{exp_name}/{weights_file}", device_map="auto")
# model = model.merge_and_unload()
model.eval()

set_seed(seed)
full_dataset = torch.utils.data.ConcatDataset([
    LlamaDataset(load_dataset(data_name, "main", split="train"), tokenizer, max_length=512),
    LlamaDataset(load_dataset(data_name, "main", split="test"), tokenizer, max_length=512)
])
total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
_, _, test_ds = random_split(full_dataset, [train_size, val_size, test_size])


# %%
def parse_correct(s):
    s_split = s.split("#")
    if len(s_split) <= 1:
        raise ValueError("Output format does not meet the requirements: ### <number>")
    pattern = re.compile(r'\d+')
    nums = pattern.findall(s_split[-1])
    if len(nums) <= 0:
        raise ValueError("Unable to parse numbers")
    return nums[0]


def get_correct_ans(s):
    try:
        return parse_correct(s)
    except ValueError as e:
        return math.nan


# %%
csv_path = f"./result/{exp_name}/eval_ans_acc.csv"
loop = tqdm(test_ds, desc="test", position=0)
for data in loop:
    new_row = {"question": None, "answer": None, "output": None, "correct": None, "output_num": None, "acc": None}

    question = data["question"]
    prompt = f"[INST] <<SYS>>\n{instruction}\n<</SYS>>\n{question} [/INST]"
    input = tokenizer(prompt, return_tensors="pt").to(model.device)

    answer = data["answer"]

    generate_ids = model.generate(
        input_ids=input['input_ids'],
        max_new_tokens=1024,
        do_sample=False,
        top_p=1,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id)

    output = tokenizer.batch_decode(generate_ids)[0]
    correct = get_correct_ans(answer)
    output_num = get_correct_ans(output)
    acc = 1 if correct == output_num else 0

    new_row["question"] = question
    new_row["answer"] = answer
    new_row["output"] = output
    new_row["correct"] = correct
    new_row["output_num"] = output_num
    new_row["acc"] = acc

    loop.set_postfix(correct=correct,
                     output_num=output_num,
                     acc=acc)
    # mean_acc=sum(exp_dict["acc"]) / len(exp_dict["acc"])

    df = pd.DataFrame([new_row])
    df.to_csv(csv_path, mode='a', header=not pd.read_csv(csv_path).shape[0])
