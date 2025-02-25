from dataset.huggingface_dataset import HuggingFaceDataset
from dataset.tokenizer_wrapper import TokenizerWrapper
from trainer.trainer import LoRATrainer

# Load Dataset
dataset_name = "medalpaca/medical_meadow_medical_flashcards"
dataset = HuggingFaceDataset(dataset_name)
dataset.load_data()

# Load Tokenizer
tokenizer = TokenizerWrapper("meta-llama/Llama-3.2-1B").tokenizer

# Preprocess Dataset
dataset.preprocess_data(tokenizer)
dataset.split_data()

# Select Small Sample for Training
small_train = dataset.dataset["train"].select(range(1000))  # First 3000 samples
small_val = dataset.dataset["test"].select(range(200))  # First 600 samples

# Initialize LoRA Trainer
trainer = LoRATrainer(
    model_id="meta-llama/Llama-3.2-1B",
    dataset={"train": small_train, "test": small_val},  # Pass small sample
    tokenizer=tokenizer
)

# Train Model on Small Sample
trainer.train(num_train_epochs=1, per_device_batch_size=4)
