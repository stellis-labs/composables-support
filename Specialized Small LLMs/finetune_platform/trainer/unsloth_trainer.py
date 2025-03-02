import torch
from unsloth import FastLora
from transformers import TrainingArguments, Trainer
from models.model_factory import get_model

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
        """Fine-tunes the model using Unsloth's LoRA trainer."""

        # Load model using Unsloth
        print("Loading model with Unsloth...")
        model = get_model(self.model_name, self.model_id)
        model.load_model()

        # Apply LoRA with Unsloth
        print("Applying LoRA with Unsloth...")
        model.model = FastLora.apply_lora(
            model.model,
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Tokenized dataset
        train_dataset = self.dataset["train"]
        val_dataset = self.dataset["test"]

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            logging_steps=10,
            learning_rate=5e-4,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            fp16=True,
            push_to_hub=False
        )

        # Trainer Setup
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  
            tokenizer=self.tokenizer
        )

        # Start Training
        print("Starting LoRA fine-tuning with Unsloth...")
        trainer.train()

        # Save Model
        model.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"\nUnsloth fine-tuned model saved at: {self.output_dir}")
