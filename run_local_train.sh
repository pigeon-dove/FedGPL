#!/bin/bash
python main.py --batch_size=2 --grad_accum_steps=16 --max_steps=20000 --val_steps=500 --lr=1e-3
