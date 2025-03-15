import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .base_model import BaseModel

class LLaMAModel(BaseModel):
    """LLaMA 3.2-1B model with LoRA fine-tuning support."""

    def load_model(self):
        """Loads LLaMA model based on Hugging Face or Unsloth."""
        print(f"Loading LLaMA model: {self.model_id}")

        if self.use_unsloth:
            print("Using Unsloth for model loading...")
            self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
                model_name=self.model_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                load_in_4bit=True,
                torch_dtype="auto",
                device_map="auto"
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def apply_lora(self, r=16, lora_alpha=16, lora_dropout=0.05):
        """Applies LoRA configuration."""
        print("Applying LoRA configuration...")

        if self.use_unsloth:
            print("Applying LoRA using Unsloth...")
            self.model = unsloth.FastLanguageModel.get_peft_model(
                self.model,
                r=r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth"
            )
        else:
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
