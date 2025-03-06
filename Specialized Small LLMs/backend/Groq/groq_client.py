import time
from groq import Groq

class GroqClient:
    def __init__(self, api_key, model_name):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
    
    def generate_completion(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.9
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        raise Exception("Max retries exceeded")