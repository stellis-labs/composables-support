from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from models.model_factory import get_model

class LoRATrainer:
    """Handles LoRA fine-tuning for Hugging Face models."""

    def __init__(self, model_name, model_id, dataset, tokenizer, output_dir="./fine-tuned-model"):
        self.model_name = model_name  # "llama" or "qwen"
        self.model_id = model_id
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def train(self, num_train_epochs=1, per_device_batch_size=4):
        """Fine-tunes the model using LoRA."""
        model = get_model(self.model_name, self.model_id)  # Select model dynamically
        model.load_model()
        model.apply_lora()

        # Tokenized dataset
        train_dataset = self.dataset["train"]
        val_dataset = self.dataset["test"]

        # Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=model.model, 
            padding="longest"
        )

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
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Start Training
        print("Starting LoRA fine-tuning...")
        trainer.train()

        # Save Model
        model.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Print Training Logs
        print("\nTraining Log History:")
        print(trainer.state.log_history)
