from unsloth import is_bfloat16_supported

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 5e-6
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.weight_decay = 0.1
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "cosine"
        self.optim = "adamw_8bit"
        self.logging_steps = 1
        self.bf16 = is_bfloat16_supported()
        self.fp16 = not is_bfloat16_supported()
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.num_generations = 8
        self.max_prompt_length = 256
        self.max_completion_length = 200
        self.max_steps = 250
        self.save_steps = 250
        self.max_grad_norm = 0.1
        self.output_dir = "outputs"