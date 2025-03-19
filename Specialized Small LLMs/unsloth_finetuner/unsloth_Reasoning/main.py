from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.dataset_processor import DatasetProcessor
from data.reward_functions import RewardFunctions
from models.model_loader import ModelLoader
from training.grpo_trainer import GRPOTrainerSetup
from inference.generator import ResponseGenerator

def main():
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Load and prepare model
    loader = ModelLoader(model_config)
    model, tokenizer = loader.load_model()
    model = loader.apply_lora(model)
    
    # Prepare dataset
    processor = DatasetProcessor()
    dataset = processor.load_gsm8k()
    
    # Setup trainer
    reward_funcs = [
        RewardFunctions.xml_count_reward,
        RewardFunctions.soft_format_reward,
        RewardFunctions.strict_format_reward,
        RewardFunctions.int_reward,
        RewardFunctions.correctness_reward,
    ]
    
    trainer_setup = GRPOTrainerSetup(
        model, tokenizer, dataset, training_config, reward_funcs
    )
    trainer = trainer_setup.create_trainer()
    
    # Train
    trainer.train()
    
    # Generate example
    generator = ResponseGenerator(model, tokenizer)
    messages = [
        {"role": "user", "content": "How many r's are in strawberry?"}
    ]
    print(generator.generate(messages))

if __name__ == "__main__":
    main()