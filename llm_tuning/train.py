import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_tuning.dataset import inf_dl_gen


def data_to_device(*args, device="cuda:0"):
    res = []
    for arg in args:
        res.append(arg.to(device))
    return res


class LlmTrainer:

    def __init__(self, model, tokenizer, train_ds, val_dl, config):
        self.model = model
        self.inf_train_iter = iter(inf_dl_gen(DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)))
        self.tokenizer = tokenizer
        self.val_dl = val_dl
        self.config = config

    def train(self):
        writer = SummaryWriter(f"./result/{self.config.exp_name}/logs")
        writer.add_hparams(self.config.__dict__, {})

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=self.config.max_steps // self.config.grad_accum_steps,
                                                   eta_min=self.config.lr / 10)
        loss_list, acc_list = [], []

        loop = tqdm(range(self.config.max_steps))
        for step in loop:
            self.model.train()
            batch_data = next(self.inf_train_iter)
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

            loss_list.append(loss.item())
            acc_list.append(acc.item())

            if (step + 1) % self.config.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                mean_loss = sum(loss_list) / len(loss_list)
                mean_acc = sum(acc_list) / len(acc_list)
                writer.add_scalar("Loss/train", mean_loss, step)
                writer.add_scalar("Accuracy/train", mean_acc, step)
                loop.set_postfix(mean_acc=mean_acc, mean_loss=mean_loss)
                loss_list.clear()
                acc_list.clear()

            if (step + 1) % self.config.val_steps == 0 or (step + 1) == self.config.max_steps:
                val_loss, val_acc = self.validate()
                writer.add_scalar("Loss/validate", val_loss, step)
                writer.add_scalar("Accuracy/validate", val_acc, step)
                self.model.save_pretrained(f"./result/{self.config.exp_name}/weights-{step + 1}")

        writer.close()

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
        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)
