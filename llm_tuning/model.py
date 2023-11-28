import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, LlamaTokenizer


def get_4bit_model(model_name, token):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        token=token,
    )

    return model


def get_lora_model(model, lora_r=16):
    lora_config = LoraConfig(r=lora_r,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             bias="none",
                             task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)
    return model


def get_tokenizer(model_name, token):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.cls_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    return tokenizer
