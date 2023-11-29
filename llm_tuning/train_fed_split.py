import collections
import copy
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


def data_to_device(*args, device="cuda"):
    res = []
    for arg in args:
        res.append(arg.to(device))
    return res


class FedClient:

    def __init__(self, train_ds, model, tokenizer, batch_size, accum_steps, batch_num, epoch, client_lr):
        self.cycled_dl = cycle(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False))
        self.model = model
        self.tokenizer = tokenizer
        self.accum_steps = accum_steps
        self.batch_num = batch_num
        self.epoch = epoch
        self.client_lr = client_lr

    def local_train(self, lora_weight, i):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.SGD(self.model.base_model.model.model.layers[2 * i:2 * i + 2].parameters(),
                                    lr=self.client_lr)
        train_dl = [next(self.cycled_dl) for _ in range(self.batch_num)]
        acc_sum, loss_sum = 0, 0

        for e in range(self.epoch):
            optimizer.zero_grad()
            for i, batch_data in enumerate(train_dl):
                input_ids, attention_mask, label_mask = data_to_device(batch_data["input_ids"],
                                                                       batch_data["attention_mask"],
                                                                       batch_data["label_mask"],
                                                                       device=self.model.device)
                output = self.model.forward(input_ids, attention_mask)
                shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.tokenizer.vocab_size)
                shift_labels = input_ids[..., 1:].contiguous().view(-1)
                shift_mask = label_mask[..., 1:].contiguous().view(-1)

                loss = (criterion(shift_logits, shift_labels) * shift_mask).sum() / (shift_mask.sum() + 1e-9)
                acc = ((shift_logits.argmax(dim=-1) == shift_labels) * shift_mask).sum() / (shift_mask.sum() + 1e-9)
                loss.backward()

                acc_sum += acc.item()
                loss_sum += loss.item()

                if (i + 1) % self.accum_steps == 0 or (i + 1) == len(train_dl):
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
        base_size = len(train_ds) // config.client_num
        remainder = len(train_ds) % config.client_num
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(config.client_num)]
        train_ds_split = random_split(train_ds, split_sizes)

        for client_ds in train_ds_split:
            client = FedClient(client_ds, self.model, tokenizer, config.batch_size, config.grad_accum_steps,
                               config.client_batch_per_step, config.client_epoch, config.client_lr)
            self.client_list.append(client)

    def train(self):
        writer = SummaryWriter(f"./result/{self.config.exp_name}/logs", flush_secs=10)
        writer.add_hparams(self.config.__dict__, {})

        server_optimizer = [
            torch.optim.AdamW(self.model.base_model.model.model.layers[2 * i:2 * i + 2].parameters(), lr=self.config.lr)
            for i in range(16)
        ]

        loop = tqdm(range(self.config.max_steps), ncols=100)
        for step in loop:
            self.model.train()
            grad_collector = MeanGradCollector()
            acc_sum, loss_sum = 0, 0
            self.require_grad_all()
            sel_layer, sel_grad_mean_list = self.calc_select_layer()
            self.require_grad(sel_layer)
            lora_weight = save_lora_weight(self.model, self.lora_module_name)

            for client in random.sample(self.client_list, self.config.client_num_per_step):
                grad, acc, loss = client.local_train(lora_weight, sel_layer)
                load_lora_weight(self.model, lora_weight)  # restore the model weights
                grad_collector.add(grad)
                acc_sum += acc
                loss_sum += loss
            mean_grad = grad_collector.mean_grad(lr=self.config.client_lr)
            mean_acc, mean_loss = acc_sum / self.config.client_num_per_step, loss_sum / self.config.client_num_per_step

            server_optimizer[sel_layer].zero_grad()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad = mean_grad[n]
            server_optimizer[sel_layer].step()

            writer.add_scalar("accuracy/train", mean_acc, step)
            writer.add_scalar("loss/train", mean_loss, step)
            writer.add_scalar("selected_layer/train", sel_layer, step)
            for i, g_mean in enumerate(sel_grad_mean_list):
                writer.add_scalar(f"layer_grad_{i}/select", g_mean, step)
            loop.set_postfix(mean_acc=mean_acc, mean_loss=mean_loss, sel_layer=sel_layer)

            if (step + 1) % self.config.val_steps == 0 or (step + 1) == self.config.max_steps:
                val_loss, val_acc = self.validate()
                writer.add_scalar("loss/validate", val_loss, step)
                writer.add_scalar("accuracy/validate", val_acc, step)
                self.model.save_pretrained(f"./result/{self.config.exp_name}/weights-{step + 1}")
        writer.close()

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
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(self.val_dl, desc="validate", position=0)):
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
        layer_grad_mean_list = []
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.model.zero_grad()
        val_iter = iter(self.val_dl)
        for i in range(4):
            batch_data = next(val_iter)
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
        for i in range(16):
            g_mean, num = 0, 0
            for p in self.model.base_model.model.model.layers[2 * i:2 * i + 2].parameters():
                if p.requires_grad:
                    g_mean += p.grad.abs().mean().item()
                    num += 1
            g_mean = g_mean / num
            layer_grad_mean_list.append(g_mean)
        sel_layer = np.argmax(layer_grad_mean_list)
        self.model.zero_grad()
        return sel_layer, layer_grad_mean_list

    def require_grad(self, i):
        for n, p in self.model.base_model.model.model.layers[2 * i:2 * i + 2].named_parameters():
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
