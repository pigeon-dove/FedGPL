import torch
from torch.utils.data import Dataset, Subset
from transformers import LlamaTokenizer


def inf_dl_gen(dataloader):
    while True:
        for data in dataloader:
            yield data


def get_prompt(instruction, question, model_name):
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        question_prompt = f"<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n{question} [/INST] "
    elif model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        question_prompt = f"<|system|>\n{instruction}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
    elif model_name == "bigscience/bloom-3b":
        question_prompt = f"{instruction}\nQuestion: {question}\nAnswer: "
    else:
        raise f"model {model_name} not found"
    return question_prompt


class Gsm8kDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: LlamaTokenizer, max_length, model_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # instruction = "Answer the following maths questions."
        instruction = "Please solve the following math problem and provide the answer in the format '### <number>' at the end."
        question = data["question"]
        answer = data["answer"]

        question_prompt = get_prompt(instruction, question, self.model_name)
        answer_prompt = f"{answer}</s>"

        question_token = self.tokenizer(question_prompt, return_length=True)
        answer_token = self.tokenizer(answer_prompt, return_length=True)

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = question_token["attention_mask"] + answer_token["attention_mask"]
        label_mask = [0] * len(question_token["input_ids"]) + [1] * len(answer_token["input_ids"])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_mask = label_mask[:self.max_length]
        else:
            res_size = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * res_size
            attention_mask = attention_mask + [0] * res_size
            label_mask = label_mask + [0] * res_size

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "origin_token_length": question_token.length + answer_token.length,
            "question": question,
            "answer": answer
        }


class MathDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: LlamaTokenizer, max_length, model_name):
        self.dataset = [data for data in dataset if data["topic;"] == "Algebra"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        instruction = "Please solve the following math problem step by step."
        question = data["message_1"]
        answer = data["message_2"]

        question_prompt = get_prompt(instruction, question, self.model_name)
        answer_prompt = f"{answer} </s>"

        question_token = self.tokenizer(question_prompt, return_length=True)
        answer_token = self.tokenizer(answer_prompt, return_length=True)

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = question_token["attention_mask"] + answer_token["attention_mask"]
        label_mask = [0] * len(question_token["input_ids"]) + [1] * len(answer_token["input_ids"])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_mask = label_mask[:self.max_length]
        else:
            res_size = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * res_size
            attention_mask = attention_mask + [0] * res_size
            label_mask = label_mask + [0] * res_size

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "origin_token_length": question_token.length + answer_token.length,
            "question": question,
            "answer": answer
        }


class SVAMPDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: LlamaTokenizer, max_length, model_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        instruction = "Solve the following math problem, giving the formula and answer directly."
        question = data["Body"] + data["Question"]
        answer = data["Equation"] + str(data["Answer"])

        question_prompt = get_prompt(instruction, question, self.model_name)
        answer_prompt = f"{answer} </s>"

        question_token = self.tokenizer(question_prompt, return_length=True)
        answer_token = self.tokenizer(answer_prompt, return_length=True)

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = question_token["attention_mask"] + answer_token["attention_mask"]
        label_mask = [0] * len(question_token["input_ids"]) + [1] * len(answer_token["input_ids"])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_mask = label_mask[:self.max_length]
        else:
            res_size = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * res_size
            attention_mask = attention_mask + [0] * res_size
            label_mask = label_mask + [0] * res_size

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "origin_token_length": question_token.length + answer_token.length,
            "question": question,
            "answer": answer
        }


class MultiArithDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: LlamaTokenizer, max_length, model_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        instruction = "Solve the following math problem, giving the answer number directly."
        question = data["question"]
        answer = data["final_ans"]

        question_prompt = get_prompt(instruction, question, self.model_name)
        answer_prompt = f"{answer} </s>"

        question_token = self.tokenizer(question_prompt, return_length=True)
        answer_token = self.tokenizer(answer_prompt, return_length=True)

        input_ids = question_token["input_ids"] + answer_token["input_ids"]
        attention_mask = question_token["attention_mask"] + answer_token["attention_mask"]
        label_mask = [0] * len(question_token["input_ids"]) + [1] * len(answer_token["input_ids"])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_mask = label_mask[:self.max_length]
        else:
            res_size = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * res_size
            attention_mask = attention_mask + [0] * res_size
            label_mask = label_mask + [0] * res_size

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "origin_token_length": question_token.length + answer_token.length,
            "question": question,
            "answer": answer
        }
