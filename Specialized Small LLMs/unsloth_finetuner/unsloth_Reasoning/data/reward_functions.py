import re

class RewardFunctions:
    @staticmethod
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        return answer.strip()

    @staticmethod
    def correctness_reward(prompts, completions, answer):
        responses = [c[0]['content'] for c in completions]
        extracted = [RewardFunctions.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    @staticmethod
    def int_reward(completions):
        responses = [c[0]['content'] for c in completions]
        extracted = [RewardFunctions.extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    @staticmethod
    def strict_format_reward(completions):
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        return [0.5 if re.match(pattern, c[0]["content"]) else 0.0 for c in completions]

    @staticmethod
    def soft_format_reward(completions):
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        return [0.5 if re.match(pattern, c[0]["content"]) else 0.0 for c in completions]

    @staticmethod
    def xml_count_reward(completions):
        def count_xml(text):
            count = 0.0
            if text.count("<reasoning>\n") == 1: count += 0.125
            if text.count("\n</reasoning>\n") == 1: count += 0.125
            if text.count("\n<answer>\n") == 1: 
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            return count
        return [count_xml(c[0]["content"]) for c in completions]