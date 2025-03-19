from vllm import SamplingParams

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = 1024,
        )

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        return self.model.fast_generate(
            [text],
            sampling_params = self.sampling_params,
            lora_request = None,
        )[0].outputs[0].text