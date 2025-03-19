class ModelSaver:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def save_locally(self):
        cfg = self.config.saving_config
        if cfg["local_saves"]["merged_16bit"]:
            self.model.save_pretrained_merged(
                cfg["local_path"], 
                self.tokenizer, 
                save_method="merged_16bit"
            )
        if cfg["local_saves"]["merged_4bit"]:
            self.model.save_pretrained_merged(
                cfg["local_path"], 
                self.tokenizer, 
                save_method="merged_4bit"
            )
        if cfg["local_saves"]["lora"]:
            self.model.save_pretrained_merged(
                cfg["local_path"], 
                self.tokenizer, 
                save_method="lora"
            )
        if any([cfg["local_saves"]["gguf_q8"], 
               cfg["local_saves"]["gguf_16bit"], 
               cfg["local_saves"]["gguf_q4"]]):
            self._save_gguf_locally()

    def push_to_hub(self):
        cfg = self.config.saving_config
        if any(cfg["hub_saves"].values()):
            self._push_merged_formats()
            self._push_gguf_formats()

    def _save_gguf_locally(self):
        cfg = self.config.saving_config
        if cfg["local_saves"]["gguf_q8"]:
            self.model.save_pretrained_gguf(cfg["local_path"], self.tokenizer)
        if cfg["local_saves"]["gguf_16bit"]:
            self.model.save_pretrained_gguf(
                cfg["local_path"], 
                self.tokenizer, 
                quantization_method="f16"
            )
        if cfg["local_saves"]["gguf_q4"]:
            self.model.save_pretrained_gguf(
                cfg["local_path"], 
                self.tokenizer, 
                quantization_method="q4_k_m"
            )

    def _push_merged_formats(self):
        cfg = self.config.saving_config
        if cfg["hub_saves"]["merged_16bit"]:
            self.model.push_to_hub_merged(
                cfg["hub_path"], 
                tokenizer=self.tokenizer,
                save_method="merged_16bit",
                token=cfg["hf_token"]
            )
        if cfg["hub_saves"]["merged_4bit"]:
            self.model.push_to_hub_merged(
                cfg["hub_path"], 
                tokenizer=self.tokenizer,
                save_method="merged_4bit",
                token=cfg["hf_token"]
            )
        if cfg["hub_saves"]["lora"]:
            self.model.push_to_hub_merged(
                cfg["hub_path"], 
                tokenizer=self.tokenizer,
                save_method="lora",
                token=cfg["hf_token"]
            )

    def _push_gguf_formats(self):
        cfg = self.config.saving_config
        if any([cfg["hub_saves"]["gguf_q8"], 
               cfg["hub_saves"]["gguf_16bit"], 
               cfg["hub_saves"]["gguf_q4"]]):
            self.model.push_to_hub_gguf(
                cfg["hub_path"],
                self.tokenizer,
                quantization_method=cfg["gguf_quantizations"],
                token=cfg["hf_token"]
            )