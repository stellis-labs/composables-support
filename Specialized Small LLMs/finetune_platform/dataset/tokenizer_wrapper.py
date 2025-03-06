from transformers import AutoTokenizer

class TokenizerWrapper:
    """Handles tokenization for multiple models"""

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fix for LLaMA padding issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_text(self, texts):
        """Tokenizes text for fine-tuning"""
        return self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
