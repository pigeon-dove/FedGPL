# %%
import argparse
import math
import os
import re
import torch
from datasets import load_dataset
from peft import PeftModel
from tensorboardX import SummaryWriter
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM

from llm_tuning.dataset import Gsm8kDataset
from llm_tuning.utils import set_seed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"
data_name = "gsm8k"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--weight_path", default="", type=str)
    return parser.parse_args()


config = parse_args()

exp_name = config.exp_name
weight_path = config.weight_path
seed = 3407
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

if weight_path != "":
    model = PeftModel.from_pretrained(model, weight_path, device_map="auto")
    model = model.merge_and_unload()
model.eval()

set_seed(seed)
full_dataset = torch.utils.data.ConcatDataset([
    Gsm8kDataset(load_dataset(data_name, "main", split="train"), tokenizer, max_length=512),
    Gsm8kDataset(load_dataset(data_name, "main", split="test"), tokenizer, max_length=512)
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
writer = SummaryWriter(f"./result/ans_acc/{exp_name}/logs", flush_secs=30)

loop = tqdm(test_ds, desc="test", position=0)
for i, data in enumerate(loop):
    question = data["question"]
    answer = data["answer"]

    prompt = f"[INST] <<SYS>>\n{instruction}\n<</SYS>>\n{question} [/INST]"
    input = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    writer.add_text("question", question, i)
    writer.add_text("answer", answer, i)
    writer.add_text("output", output, i)
    writer.add_text("correct", str(correct), i)
    writer.add_text("output_num", str(output_num), i)
    writer.add_scalar("acc", acc, i)
    writer.flush()

    loop.set_postfix(correct=correct,
                     output_num=output_num,
                     acc=acc)
