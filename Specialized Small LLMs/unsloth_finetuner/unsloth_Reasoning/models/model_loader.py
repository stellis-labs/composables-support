from unsloth import FastLanguageModel

class ModelLoader:
    def __init__(self, config):
        self.config = config
        
    def load_model(self):
        return FastLanguageModel.from_pretrained(
            model_name = self.config.model_name,
            max_seq_length = self.config.max_seq_length,
            load_in_4bit = self.config.load_in_4bit,
            fast_inference = self.config.fast_inference,
            max_lora_rank = self.config.lora_rank,
            gpu_memory_utilization = self.config.gpu_memory_utilization,
        )

    def apply_lora(self, model):
        return FastLanguageModel.get_peft_model(
            model,
            r = self.config.lora_rank,
            target_modules = self.config.target_modules,
            lora_alpha = self.config.lora_rank,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )