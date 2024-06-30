import torch
from peft import LoraConfig, get_peft_model, PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig, \
    PromptEncoderReparameterizationType
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer


def get_4bit_model(model_name, token, device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device,
        token=token,
    )

    return model


def get_lora_model(model, lora_r=16):
    config = LoraConfig(r=lora_r,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             bias="none",
                             task_type="CAUSAL_LM")

    model = get_peft_model(model, config)
    return model


def get_ptuning_model(model):
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.num_layers

    config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=num_layers,
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=768,
    )

    model = get_peft_model(model, config)
    return model


def get_prompt_model(model):
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.num_layers

    config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type="CAUSAL_LM",
        num_virtual_tokens=40,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=num_layers
    )
    model = get_peft_model(model, config)
    return model


def get_tokenizer(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.cls_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    return tokenizer
