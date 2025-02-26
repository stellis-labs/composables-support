from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class BaseModel(ABC):
    """Abstract base class for LoRA fine-tuning models."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load_model(self):
        """Loads the base model for fine-tuning."""
        pass

    @abstractmethod
    def apply_lora(self):
        """Applies LoRA configuration to the model."""
        pass

    def save_model(self, path):
        """Saves the fine-tuned model."""
        self.model.save_pretrained(path)
