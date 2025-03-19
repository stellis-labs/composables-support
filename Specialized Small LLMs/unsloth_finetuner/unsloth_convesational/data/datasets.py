from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

def load_and_process_dataset(dataset_name, split, tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    dataset = load_dataset(dataset_name, split=split)
    dataset = standardize_sharegpt(dataset)
    return dataset.map(formatting_prompts_func, batched=True)