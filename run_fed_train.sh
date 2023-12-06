#!/bin/bash
python main.py --device="cuda:0" --train_mode="fedSplit" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=4000 --val_steps=50 --lr=1e-3 --client_lr=1e-4
