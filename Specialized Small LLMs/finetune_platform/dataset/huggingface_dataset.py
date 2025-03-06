from datasets import load_dataset
from .base_dataset import BaseDataset
from .preprocessing import preprocess_text

class HuggingFaceDataset(BaseDataset):
    """Loads and processes Hugging Face datasets dynamically"""

    def load_data(self):
        """Loads dataset from Hugging Face"""
        print(f"Loading dataset from Hugging Face: {self.dataset_path}")
        self.dataset = load_dataset(self.dataset_path)  # Now dynamically loads dataset
        print("Dataset loaded successfully.")

    def preprocess_data(self, tokenizer):
        """Processes dataset into structured user-assistant format"""

        def tokenize_function(examples):
            formatted_texts = preprocess_text(examples)  # Formatting to `<|user|>...<|assistant|>`
            tokenized = tokenizer(
                formatted_texts,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        # Apply tokenization & remove raw columns
        print("Tokenizing dataset...")
        self.dataset = self.dataset.map(tokenize_function, batched=True, remove_columns=self.dataset["train"].column_names)
        print("Dataset tokenization completed.")
