#!/bin/bash
#python main.py --device="cuda:0" --train_mode="fedGradFocus" --max_layer_num=6 --min_layer_num=2 --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=5e-4 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fedGradFocus" --max_layer_num=3 --min_layer_num=1 --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=5e-4 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fed" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=5e-4 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fed" --fed_alg="FedProx" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=1 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fed" --fed_alg="FedAVG" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=1 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fed" --peft="p-tuning" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=5e-4 --client_lr=1e-4
#python main.py --device="cuda:0" --train_mode="fed" --peft="prompt-tuning" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=2000 --val_steps=20 --lr=5e-4 --client_lr=1e-4



python main.py \
--device="cuda:0" --train_mode="fed" --peft="lora" --data_name="camel-ai/math" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 \
--client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4

python main.py \
--device="cuda:0" --train_mode="fedGradFocus" --max_layer_num=6 --min_layer_num=2 --peft="lora" --data_name="camel-ai/math" \
--fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 \
--grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4

python main.py \
--device="cuda:0" --train_mode="fedGradFocus" --max_layer_num=3 --min_layer_num=1 --peft="lora" --data_name="camel-ai/math" \
--fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 --client_epoch=1 --batch_size=2 --client_batch_per_step=8 \
--grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4

python main.py \
--device="cuda:0" --train_mode="fed" --peft="p-tuning" --data_name="camel-ai/math" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 \
--client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4

python main.py \
--device="cuda:0" --train_mode="fed" --peft="prompt-tuning" --data_name="camel-ai/math" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 \
--client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4

python main.py \
--device="cuda:0" --train_mode="fed" --peft="lora" --data_name="camel-ai/math" --fed_alg="FedAVG" --client_num=50 --client_num_per_step=4 \
--client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4