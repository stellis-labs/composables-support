from trl import GRPOConfig, GRPOTrainer

class GRPOTrainerSetup:
    def __init__(self, model, tokenizer, dataset, training_config, reward_functions):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = training_config
        self.reward_funcs = reward_functions

    def create_trainer(self):
        return GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = self.reward_funcs,
            args = GRPOConfig(**self.config.__dict__),
            train_dataset = self.dataset,
        )