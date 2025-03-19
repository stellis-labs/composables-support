class ModelConfig:
    def __init__(self):
        self.model_name = "unsloth/Llama-3.2-3B-Instruct"
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.lora_config = {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 16,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407,
            "use_rslora": False,
            "loftq_config": None,
        }
        self.saving_config = {
            "save_methods": {
                "merged_16bit": True,
                "merged_4bit": False,
                "lora": True,
                "gguf_8bit": False,
                "gguf_16bit": False,
                "gguf_q4": True,
            },
            "save_path": "trained_model",
            "hf_push": False,
            "hf_token": None,
        }
        self.training_args = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 60,
            "learning_rate": 2e-4,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": "outputs",
        }