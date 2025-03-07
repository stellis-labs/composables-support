import unsloth
from transformers import AutoTokenizer
import torch
from .base_inference import BaseInference

class LLaMAUnslothInference(BaseInference):
    """Handles inference for fine-tuned LLaMA models using Unsloth."""

    def load_model(self):
        """Loads the fine-tuned model and tokenizer using Unsloth."""
        print(f"Loading fine-tuned LLaMA (Unsloth) model from: {self.model_path}")

        # Load model using Unsloth's FastLanguageModel
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            self.model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        print("Unsloth model loaded successfully.")

    def generate_response(self, user_query, max_length=256):
        """Generates a response using the fine-tuned Unsloth LLaMA chatbot."""

        prompt = f"<|user|>\n{user_query}\n\n<|assistant|>\n"

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
