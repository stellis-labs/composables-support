from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .base_model import BaseModel

class LLaMAModel(BaseModel):
    """LLaMA 3.2-1B model with LoRA fine-tuning support."""

    def load_model(self):
        """Loads LLaMA model in 4-bit mode."""
        print(f"Loading LLaMA model: {self.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_4bit=True,
            torch_dtype="auto",
            device_map="auto"
        ).to(self.device)

    def apply_lora(self, r=8, lora_alpha=16, lora_dropout=0.05):
        """Applies LoRA to the model."""
        print("Applying LoRA configuration...")
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.gradient_checkpointing_enable()
