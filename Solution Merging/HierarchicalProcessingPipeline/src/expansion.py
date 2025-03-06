from config.settings import client

MODEL_NAME_DSLLAMA70B = "deepseek-r1-distill-llama-70b"
MODEL_NAME_DSQWEN32B = "deepseek-r1-distill-qwen-32b"
MODEL_NAME_QWEN32B = "qwen-2.5-32b"

def expand_summary(summary, original_chunks):
    """Expands the summary by reintegrating critical details using Groq LLM."""
    response = client.chat.completions.create(
        model=MODEL_NAME_DSLLAMA70B,
        messages=[
            {"role": "system", "content": "Expand this summary into a structured report, reintegrating technical details."},
            {"role": "user", "content": summary + "\n\nOriginal Data:\n" + " ".join(original_chunks)}
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95
    )
    return response.choices[0].message.content
