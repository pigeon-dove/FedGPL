#!/bin/bash
python main.py --batch_size=2 --grad_accum_steps=8 --max_steps=20000 --val_steps=500 --lr=5e-4
