#!/bin/bash
#python main.py --client_num=50 --client_num_per_step=4 --client_epoch=1 --client_batch_per_step=8 --batch_size=2 --grad_accum_steps=2 --max_steps=4000 --val_steps=100 --lr=5e-4 --client_lr=3e-3
python main.py --client_num=50 --client_num_per_step=4 --client_epoch=1 --client_batch_per_step=8 --batch_size=2 --grad_accum_steps=2 --max_steps=6000 --val_steps=100 --lr=1e-3 --client_lr=3e-3