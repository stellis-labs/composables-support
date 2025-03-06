from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class BaseInference(ABC):
    """Abstract base class for inference with fine-tuned models."""

    def __init__(self, model_path, base_model_id):
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    @abstractmethod
    def load_model(self):
        """Loads the fine-tuned model and tokenizer."""
        pass

    @abstractmethod
    def generate_response(self, user_query, max_length=256):
        """Generates a response based on user query."""
        pass
