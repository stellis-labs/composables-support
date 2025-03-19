from unsloth import FastLanguageModel

def load_model(config):
    return FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

def apply_lora(model, lora_config):
    return FastLanguageModel.get_peft_model(model, **lora_config)