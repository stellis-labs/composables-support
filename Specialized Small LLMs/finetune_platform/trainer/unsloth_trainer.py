import unsloth
from transformers import TrainingArguments
from models.model_factory import get_model
from trl import SFTTrainer
import torch

class UnslothTrainer:
    """Handles Unsloth-based LoRA fine-tuning for Hugging Face models."""

    def __init__(self, model_name, model_id, dataset, tokenizer, output_dir):
        self.model_name = model_name
        self.model_id = model_id
        self.dataset = dataset
        self.tokenizer = tokenizer  
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, num_train_epochs=1, per_device_batch_size=4):
        """Fine-tunes the model using TRL’s `SFTTrainer`."""

        # Load model using Unsloth
        print("Loading model with Unsloth...")
        model = get_model(self.model_name, self.model_id, use_unsloth=True)
        model.load_model()
        model.apply_lora()  

        use_bf16 = True  

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            max_steps=60,  
            learning_rate=2e-4,
            bf16=use_bf16, 
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01
        )

        # Use `SFTTrainer`
        trainer = SFTTrainer(
            model=model.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset["train"],
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args
        )

        # Start Training
        print("Starting LoRA fine-tuning with Unsloth...")
        trainer.train()

        # Save Model
        model.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"\nUnsloth fine-tuned model saved at: {self.output_dir}")
