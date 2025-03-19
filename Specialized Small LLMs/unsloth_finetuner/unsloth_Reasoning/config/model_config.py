class ModelConfig:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.max_seq_length = 1024
        self.lora_rank = 64
        self.load_in_4bit = True
        self.fast_inference = True
        self.gpu_memory_utilization = 0.5
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        self.saving_config = {
            "local_saves": {
                "merged_16bit": True,
                "merged_4bit": False,
                "lora": False,
                "gguf_q8": False,
                "gguf_16bit": False,
                "gguf_q4": True,
            },
            "hub_saves": {
                "merged_16bit": False,
                "merged_4bit": False,
                "lora": False,
                "gguf_q8": False,
                "gguf_16bit": False,
                "gguf_q4": False,
            },
            "hub_path": "your_username/your_model",
            "local_path": "saved_models",
            "hf_token": None,
            "gguf_quantizations": ["q4_k_m", "q8_0", "q5_k_m"],
        }