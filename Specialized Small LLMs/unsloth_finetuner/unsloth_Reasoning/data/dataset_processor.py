from datasets import load_dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

class DatasetProcessor:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        
    def extract_hash_answer(self, text: str) -> str | None:
        if "####" not in text: return None
        return text.split("####")[1].strip()

    def load_gsm8k(self, split="train"):
        data = load_dataset('openai/gsm8k', 'main')[split]
        return data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': self.extract_hash_answer(x['answer'])
        })