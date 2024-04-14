import collections
import copy
import math
import random
from itertools import cycle

import numpy as np
import torch.distributed as dist
import torch
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from bitsandbytes import functional as bf


def data_to_device(*args, device="cuda"):
    res = []
    for arg in args:
        res.append(arg.to(device))
    return res


def get_layers(model, model_name):
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        path = "base_model.model.model.layers"
    elif model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        path = "base_model.model.layers"
    else:
        raise f"model {model_name} not found"
    return get_nested_field(model, path)


class FedClient:

    def __init__(self, train_ds, model, tokenizer, batch_size, accum_steps, batch_num, epoch, client_lr, model_name):
        self.cycled_dl = cycle(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False))
        self.model = model
        self.tokenizer = tokenizer
        self.accum_steps = accum_steps
        self.batch_num = batch_num
        self.epoch = epoch
        self.client_lr = client_lr
        self.model_name = model_name

    def local_train(self, lora_weight, select_indexes):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        selected_parameters = [param for i, layer in enumerate(get_layers(self.model, self.model_name)) if
                               i in select_indexes for param in layer.parameters()]
        optimizer = torch.optim.SGD(selected_parameters, lr=self.client_lr)

        train_dl = [next(self.cycled_dl) for _ in range(self.batch_num)]
        acc_sum, loss_sum = 0, 0

        for e in range(self.epoch):
            optimizer.zero_grad()
            for step, batch_data in enumerate(train_dl):
                input_ids, attention_mask, label_mask = data_to_device(batch_data["input_ids"],
                                                                       batch_data["attention_mask"],
                                                                       batch_data["label_mask"],
                                                                       device=self.model.device)
                output = self.model.forward(input_ids, attention_mask)
                shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.tokenizer.vocab_size)
                shift_labels = input_ids[..., 1:].contiguous().view(-1)
                shift_mask = label_mask[..., 1:].contiguous().view(-1)

                loss = (criterion(shift_logits, shift_labels) * shift_mask).sum() / shift_mask.sum()
                acc = ((shift_logits.argmax(dim=-1) == shift_labels) * shift_mask).sum() / shift_mask.sum()
                loss.backward()

                acc_sum += acc.item()
                loss_sum += loss.item()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(train_dl):
                    optimizer.step()
                    optimizer.zero_grad()

        grad = calc_grad(lora_weight, self.model)
        return grad, acc_sum / self.epoch / len(train_dl), loss_sum / self.epoch / len(train_dl)


class LlmFedSplitTrainer:

    def __init__(self, model, tokenizer, train_ds, val_dl, config):
        self.model = model.to(config.device)
        self.lora_module_name = self.get_lora_module_name()
        self.val_dl = val_dl
        self.tokenizer = tokenizer
        self.config = config
        self.client_list = []
        server_ds, client_ds = random_split(train_ds,
                                            [int(len(train_ds) * 0.05), len(train_ds) - int(len(train_ds) * 0.05)])
        self.server_dl = cycle(DataLoader(server_ds, batch_size=config.batch_size, shuffle=True))

        base_size = len(client_ds) // config.client_num
        remainder = len(client_ds) % config.client_num
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(config.client_num)]
        train_ds_split = random_split(client_ds, split_sizes)

        for client_ds in train_ds_split:
            client = FedClient(client_ds, self.model, tokenizer, config.batch_size, config.grad_accum_steps,
                               config.client_batch_per_step, config.client_epoch, config.client_lr, config.model_name)
            self.client_list.append(client)

    def train(self):
        writer = SummaryWriter(f"./result/{self.config.exp_name}/logs", flush_secs=10)
        writer.add_hparams(self.config.__dict__, {})

        server_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        val_acc_history = []

        loop = tqdm(range(self.config.max_steps), ncols=140)
        for step in loop:
            self.model.train()
            grad_collector = MeanGradCollector()
            acc_sum, loss_sum = 0, 0
            self.require_grad_all()
            sorted_indexes, layer_grad_norm_list, all_grad = self.calc_select_layer()

            select_indexes = sorted_indexes[:self.config.max_layer_num] + sorted_indexes[:-self.config.min_layer_num]

            self.require_grad(select_indexes)

            lora_weight = save_lora_weight(self.model, self.lora_module_name)

            for client in random.sample(self.client_list, self.config.client_num_per_step):
                grad, acc, loss = client.local_train(lora_weight, select_indexes)
                load_lora_weight(self.model, lora_weight)  # restore the model weights
                grad_collector.add(grad)
                acc_sum += acc
                loss_sum += loss
            mean_grad = grad_collector.mean_grad(lr=self.config.client_lr)
            mean_acc, mean_loss = acc_sum / self.config.client_num_per_step, loss_sum / self.config.client_num_per_step

            server_optimizer.zero_grad()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    all_grad[n] = mean_grad[n]

            self.require_grad_all()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad = all_grad[n]

            server_optimizer.step()

            writer.add_scalar("accuracy/train", mean_acc, step)
            writer.add_scalar("loss/train", mean_loss, step)
            writer.add_scalars("select_indexes",
                               {f"select_{i}": idx for i, idx in enumerate(select_indexes)}, step)

            writer.add_scalars("select/layer_grad_norm",
                               {f"layer_{i}": g for i, g in enumerate(layer_grad_norm_list)}, step)

            loop.set_postfix(mean_acc=mean_acc, mean_loss=mean_loss, select_indexes=select_indexes)
            if (step + 1) % self.config.val_steps == 0 or (step + 1) == self.config.max_steps:
                val_loss, val_acc = self.validate()
                writer.add_scalar("loss/validate", val_loss, step)
                writer.add_scalar("accuracy/validate", val_acc, step)
                self.model.save_pretrained(f"./result/{self.config.exp_name}/weights-{step + 1}")
                val_acc_history.append(val_acc)
                if len(val_acc_history) >= 3 and (val_acc_history[-1] < val_acc_history[-2] < val_acc_history[-3]):
                    break
            writer.flush()
        writer.close()

    def init_with_val(self):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr / 10)
        val_iter = iter(self.val_dl)

        new_dataloader = [next(val_iter) for _ in range(len(self.val_dl) // 20)]

        for e in range(1):
            loop = tqdm(new_dataloader, desc="init_with_val", position=0, ncols=140)
            for batch_data in loop:
                input_ids, attention_mask, label_mask = data_to_device(batch_data["input_ids"],
                                                                       batch_data["attention_mask"],
                                                                       batch_data["label_mask"],
                                                                       device=self.model.device)
                output = self.model.forward(input_ids, attention_mask)
                shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.tokenizer.vocab_size)
                shift_labels = input_ids[..., 1:].contiguous().view(-1)
                shift_mask = label_mask[..., 1:].contiguous().view(-1)

                loss = (criterion(shift_logits, shift_labels) * shift_mask).sum() / shift_mask.sum()
                acc = ((shift_logits.argmax(dim=-1) == shift_labels) * shift_mask).sum() / shift_mask.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loop.set_postfix(loss=loss.item(), acc=acc.item())

    def get_lora_module_name(self):
        name_list = []
        for n, _ in self.model.named_parameters():
            if "lora" in n:
                name_list.append(n)
        return name_list

    def validate(self):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        loss_list, acc_list = [], []

        loop = tqdm(self.val_dl, desc="validate", position=0, ncols=140)
        with torch.no_grad():
            for step, batch_data in enumerate(loop):
                input_ids, attention_mask, label_mask = data_to_device(batch_data["input_ids"],
                                                                       batch_data["attention_mask"],
                                                                       batch_data["label_mask"],
                                                                       device=self.model.device)

                output = self.model.forward(input_ids, attention_mask)
                shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.model.config.vocab_size)
                shift_labels = input_ids[..., 1:].contiguous().view(-1)
                shift_mask = label_mask[..., 1:].contiguous().view(-1)

                loss = (criterion(shift_logits, shift_labels) * shift_mask).sum() / shift_mask.sum()
                acc = ((shift_logits.argmax(dim=-1) == shift_labels) * shift_mask).sum() / shift_mask.sum()

                loss_list.append(loss.item())
                acc_list.append(acc.item())
        self.model.train()
        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def calc_select_layer(self):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.model.zero_grad()
        for i in range(4):
            batch_data = next(self.server_dl)
            input_ids, attention_mask, label_mask = data_to_device(batch_data["input_ids"],
                                                                   batch_data["attention_mask"],
                                                                   batch_data["label_mask"],
                                                                   device=self.model.device)
            output = self.model.forward(input_ids, attention_mask)
            shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.tokenizer.vocab_size)
            shift_labels = input_ids[..., 1:].contiguous().view(-1)
            shift_mask = label_mask[..., 1:].contiguous().view(-1)

            loss = (criterion(shift_logits, shift_labels) * shift_mask).sum() / (shift_mask.sum() + 1e-9)
            loss.backward()

        layer_grad_norm_list = []
        with torch.no_grad():
            for i, layer in enumerate(get_layers(self.model, self.config.model_name)):
                gradient_norm = []
                for name, module in layer.named_modules():
                    if name.endswith("q_proj") or name.endswith("v_proj"):
                        lora_a = module.lora_A.default.weight
                        lora_b = module.lora_B.default.weight
                        grad_w = lora_b @ lora_a.grad + lora_b.grad @ lora_a
                        weight_w = bf.dequantize_fp4(module.base_layer.weight.data,
                                                     module.base_layer.weight.quant_state) + lora_b @ lora_a + 1e-9
                        gradient_norm.append(torch.norm(grad_w / weight_w, p=1).item())
                gradient_norm = sum(gradient_norm) / len(gradient_norm)
                layer_grad_norm_list.append(gradient_norm)

        sorted_indexes = sorted(range(len(layer_grad_norm_list)), key=lambda k: layer_grad_norm_list[k], reverse=True)

        all_grad = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                all_grad[n] = p.grad.detach().clone() * 0.1

        self.model.zero_grad()
        return sorted_indexes, layer_grad_norm_list, all_grad

    def require_grad(self, idx_list):
        for n, p in self.model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
        for i in idx_list:
            for n, p in get_layers(self.model, self.config.model_name)[i].named_parameters():
                if "lora" in n:
                    p.requires_grad = True

    def require_grad_all(self):
        for n, p in self.model.named_parameters():
            if "lora" in n:
                p.requires_grad = True


def save_lora_weight(model, lora_module_name):
    lora_weight = {}
    for n in lora_module_name:
        lora_weight[n] = get_nested_field(model, n).detach().clone()
    return lora_weight


def load_lora_weight(model, lora_weight):
    for n, w in lora_weight.items():
        get_nested_field(model, n).data.copy_(w)
    model.train()


def calc_grad(lora_weight, model):
    grad = collections.OrderedDict()
    for n, p in lora_weight.items():
        new_p = get_nested_field(model, n)
        if new_p.requires_grad:
            grad[n] = (p - new_p).detach().clone()
    return grad


class MeanGradCollector:
    sum_grad = None
    total_num = 0

    def add(self, grad):
        self.total_num += 1
        if self.sum_grad is None:
            self.sum_grad = grad
        else:
            for k, v in grad.items():
                self.sum_grad[k] += v

    def mean_grad(self, lr):
        for k, v in self.sum_grad.items():
            self.sum_grad[k] = self.sum_grad[k] / self.total_num / lr
        return self.sum_grad


def get_nested_field(obj, field_path):
    fields = field_path.split('.')
    current_obj = obj
    for field in fields:
        current_obj = getattr(current_obj, field, None)
        if current_obj is None:
            return None
    return current_obj


def set_nested_field(obj, field_path, value):
    fields = field_path.split('.')
    current_obj = obj

    for field in fields[:-1]:
        current_obj = getattr(current_obj, field, None)
        if current_obj is None:
            return False

    setattr(current_obj, fields[-1], value)
    return True
