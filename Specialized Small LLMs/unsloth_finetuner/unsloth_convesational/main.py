from config.model_config import ModelConfig
from config.data_config import DataConfig
from data.datasets import load_and_process_dataset
from models.model_utils import load_model, apply_lora
from models.saving import save_model
from training.trainer import create_trainer, apply_response_training
from utils.memory import print_memory_stats
from unsloth.chat_templates import get_chat_template

def main():
    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Load and configure model
    model, tokenizer = load_model(model_config)
    model = apply_lora(model, model_config.lora_config)
    tokenizer = get_chat_template(tokenizer, data_config.chat_template)
    
    # Prepare dataset
    dataset = load_and_process_dataset(
        data_config.dataset_name, 
        data_config.split, 
        tokenizer
    )
    
    # Create and configure trainer
    trainer = create_trainer(model, tokenizer, dataset, model_config)
    trainer = apply_response_training(trainer, data_config)
    
    # Training
    print_memory_stats()
    trainer.train()
    
    # Save model
    save_model(model, tokenizer, model_config.saving_config)

if __name__ == "__main__":
    main()