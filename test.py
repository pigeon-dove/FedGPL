# %%
import math
import os
import re
import time

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
token = "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"

model_name = "meta-llama/Llama-2-7b-chat-hf"
data_name = "gsm8k"

# %%
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    # device_map={"": 0},
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

# model = PeftModel.from_pretrained(model, "./result/231030-1543/weights-11000", device_map={"": 0})
# model = PeftModel.from_pretrained(model, "./result/231101-1944/weights-10499", device_map={"": 0})
# model = PeftModel.from_pretrained(model, "./result/231102-1637/weights-14800", device_map={"": 0})
# model = PeftModel.from_pretrained(model, "./result/231103-1853/weights-25000", device_map={"": 0})
# model = PeftModel.from_pretrained(model, "./result/231107-1456/weights-18400", device_map={"": 1})
# model = PeftModel.from_pretrained(model, "./result/231108-1424/weights-11200", device_map={"": 0})
model = PeftModel.from_pretrained(model, "./result/231108-1424/weights-11200", device_map="auto")
model = model.merge_and_unload()
model.eval()

test_ds = load_dataset(data_name, "main", split="test")


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
        # print(e, end="")
        return math.nan


# %%
exp_dict = {"question": [], "answer": [], "output": [], "correct": [], "output_num": [], "acc": []}
loop = tqdm(test_ds, desc="test", position=0)
for data in loop:
    question = data["question"]
    instruction = "Please solve the following math problem and provide the answer in the format '### <number>' at the end."
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

    # generate_ids = model.generate(
    #     input_ids=input['input_ids'],
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.9,
    #     num_return_sequences=1,
    #     repetition_penalty=1.1,
    #     max_new_tokens=1024,
    #     eos_token_id=tokenizer.eos_token_id)

    output = tokenizer.batch_decode(generate_ids)[0]
    correct = get_correct_ans(answer)
    output_num = get_correct_ans(output)
    acc = 1 if correct == output_num else 0

    exp_dict["question"].append(question)
    exp_dict["answer"].append(answer)
    exp_dict["output"].append(output)
    exp_dict["correct"].append(correct)
    exp_dict["output_num"].append(output_num)
    exp_dict["acc"].append(acc)

    loop.set_postfix(correct=correct,
                     output_num=output_num,
                     acc=acc,
                     mean_acc=sum(exp_dict["acc"]) / len(exp_dict["acc"]))


filename = time.strftime("%y%m%d-%H%M", time.localtime(time.time()))
pd.DataFrame(exp_dict).to_csv(f"./result/{filename}.csv", index=True)
