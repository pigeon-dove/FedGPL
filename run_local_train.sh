#!/bin/bash
python main.py --device="cuda:0" --train_mode="local" --batch_size=2 --grad_accum_steps=16 --max_steps=12000 --val_steps=500 --lr=5e-4
