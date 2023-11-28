#!/bin/bash
python main.py --device="cuda:1" --train_mode="fed" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=6000 --val_steps=100 --lr=5e-4 --client_lr=1e-4
