def save_model(model, tokenizer, saving_config):
    save_path = saving_config["save_path"]
    methods = saving_config["save_methods"]
    
    if methods["merged_16bit"]:
        model.save_pretrained_merged(save_path, tokenizer, "merged_16bit")
    
    if methods["merged_4bit"]:
        model.save_pretrained_merged(save_path, tokenizer, "merged_4bit")
    
    if methods["lora"]:
        model.save_pretrained_merged(save_path, tokenizer, "lora")
    
    if methods["gguf_8bit"]:
        model.save_pretrained_gguf(save_path, tokenizer)
    
    if methods["gguf_16bit"]:
        model.save_pretrained_gguf(save_path, tokenizer, "f16")
    
    if methods["gguf_q4"]:
        model.save_pretrained_gguf(save_path, tokenizer, "q4_k_m")
    
    if saving_config["hf_push"]:
        model.push_to_hub_merged(
            save_path, 
            tokenizer=tokenizer,
            token=saving_config["hf_token"],
            save_method="merged_16bit"
        )