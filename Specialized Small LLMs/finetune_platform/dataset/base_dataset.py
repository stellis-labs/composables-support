from abc import ABC, abstractmethod
from datasets import DatasetDict

class BaseDataset(ABC):
    """Abstract class for handling datasets"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None

    @abstractmethod
    def load_data(self):
        """Loads dataset from Hugging Face or a local source"""
        pass

    @abstractmethod
    def preprocess_data(self, tokenizer):
        """Preprocess dataset for fine-tuning"""
        pass

    def split_data(self, test_size: float = 0.1):
        """Splits dataset into training and validation sets"""
        self.dataset = self.dataset["train"].train_test_split(test_size=test_size, seed=42)
