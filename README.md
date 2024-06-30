# FedGPL: Gradient Priority-based Federated Learning Enhancement for In-Vehicle Language Models



This repository contains the code for training models using different federated learning algorithms. Below are the descriptions of the configurable parameters.

## Parameters

> The `token` is used to access Meta's LLaMA model. You can apply for the token by visiting https://huggingface.co/meta-llama/Llama-2-7b

> When `train_mode` is set to `fedGradFocus`, the method described in the paper is used. This allows the use of federated learning based on the FedAvg, FedAdam, and FedProx algorithms while incorporating the techniques presented in the paper.

| Parameter               | Default Value                   | Type    | Choices                                                      | Description                                             |
| ----------------------- | ------------------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| `exp_name`              | `test`                          | `str`   | N/A                                                          | Experiment name.                                        |
| `device`                | `cuda:1`                        | `str`   | N/A                                                          | Device to run the training on (e.g., `cpu`, `cuda:0`).  |
| `token`                 | `""`                            | `str`   | N/A                                                          | Authentication token for accessing the llama model.     |
| `fed_alg`               | `FedAdam`                       | `str`   | `FedAdam`, `FedAVG`, `FedProx`                               | Federated learning algorithm to use.                    |
| `peft`                  | `lora`                          | `str`   | `lora`, `p-tuning`, `prompt-tuning`                          | Parameter-Efficient Fine-Tuning (PEFT) technique.       |
| `client_num`            | `50`                            | `int`   | N/A                                                          | Number of clients participating in federated learning.  |
| `client_num_per_step`   | `4`                             | `int`   | N/A                                                          | Number of clients to train per federated learning step. |
| `data_name`             | `gsm8k`                         | `str`   | `gsm8k`, `camel-ai/math`, `ChilleD/SVAMP`, `ChilleD/MultiArith` | Dataset to use for training.                            |
| `model_name`            | `meta-llama/Llama-2-7b-chat-hf` | `str`   | `meta-llama/Llama-2-7b-chat-hf`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `bigscience/bloom-3b` | Name of the model to use.                               |
| `client_epoch`          | `1`                             | `int`   | N/A                                                          | Number of epochs to train each client.                  |
| `client_batch_per_step` | `8`                             | `int`   | N/A                                                          | Number of batches each client processes per step.       |
| `batch_size`            | `2`                             | `int`   | N/A                                                          | Batch size for training.                                |
| `grad_accum_steps`      | `4`                             | `int`   | N/A                                                          | Number of gradient accumulation steps.                  |
| `max_steps`             | `6000`                          | `int`   | N/A                                                          | Maximum number of training steps.                       |
| `val_steps`             | `100`                           | `int`   | N/A                                                          | Validation steps interval.                              |
| `lr`                    | `5e-4`                          | `float` | N/A                                                          | Learning rate for training.                             |
| `client_lr`             | `1e-4`                          | `float` | N/A                                                          | Learning rate for client-side training.                 |
| `train_mode`            | `fedGradFocus`                  | `str`   | `local`, `fed`, `fedGradFocus`                               | Training mode to use.                                   |
| `max_layer_num`         | `6`                             | `int`   | N/A                                                          | Maximum number of layers to train.                      |
| `min_layer_num`         | `2`                             | `int`   | N/A                                                          | Minimum number of layers to train.                      |
| `grad_eval`             | `lora_per_l1`                   | `str`   | `l1`, `l2`, `lora_l1`, `lora_l2`, `lora_per_l1`, `lora_per_l2` | Method for gradient evaluation.                         |

## Script Example

```python
python main.py \
--device="cuda:0" --train_mode="fedGradFocus" --peft="lora" --data_name="ChilleD/MultiArith" --fed_alg="FedAdam" --client_num=50 --client_num_per_step=4 \
--client_epoch=1 --batch_size=2 --client_batch_per_step=8 --grad_accum_steps=4 --max_steps=1000 --val_steps=20 --lr=5e-4 --client_lr=1e-4 \
--exp_name="MultiArith_lora_llama7b" -token="your token here"
```

