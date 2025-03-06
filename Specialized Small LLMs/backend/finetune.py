from transformers import Trainer, TrainingArguments

class FineTuner:
    def __init__(self, model_name, dataset_path):
        # ... existing code ...
        self.model_name = model_name
        self.dataset_path = dataset_path
        # ... existing code ...

    def run(self):
        # ... existing code ...
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            # ... existing code ...
        )
        # ... existing code ...