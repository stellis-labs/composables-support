import sys
from dataset.huggingface_dataset import HuggingFaceDataset
from dataset.tokenizer_wrapper import TokenizerWrapper
from trainer.trainer import LoRATrainer
from trainer.unsloth_trainer import UnslothTrainer  # NEW

# Select Model at Runtime
model_name = input("Choose model for fine-tuning ('llama' or 'qwen'): ").strip().lower()
use_unsloth = input("Use Unsloth for fine-tuning? (Y/N): ").strip().lower() == "y"

# Define Model IDs & Save Paths
model_ids = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen": "qwen/qwen1.5-1b"
}
save_paths = {
    "llama": "./fine-tuned-llama" + ("-unsloth" if use_unsloth else ""),
    "qwen": "./fine-tuned-qwen" + ("-unsloth" if use_unsloth else "")
}

if model_name not in model_ids:
    print("Invalid model choice! Choose 'llama' or 'qwen'.")
    sys.exit(1)

# Load Dataset
dataset_name = "medalpaca/medical_meadow_medical_flashcards"
dataset = HuggingFaceDataset(dataset_name)
dataset.load_data()

# Load Tokenizer for Selected Model
tokenizer = TokenizerWrapper(model_ids[model_name]).tokenizer

# Preprocess Dataset
dataset.preprocess_data(tokenizer)
dataset.split_data()

# Select Small Sample for Training
small_train = dataset.dataset["train"].select(range(1000))  # First 1000 samples
small_val = dataset.dataset["test"].select(range(200))  # First 200 samples

# Choose Trainer (Hugging Face or Unsloth)
trainer_cls = UnslothTrainer if use_unsloth else LoRATrainer
trainer = trainer_cls(
    model_name=model_name,
    model_id=model_ids[model_name],
    dataset={"train": small_train, "test": small_val},
    tokenizer=tokenizer,
    output_dir=save_paths[model_name]
)

# Train Model
trainer.train(num_train_epochs=1, per_device_batch_size=4)
