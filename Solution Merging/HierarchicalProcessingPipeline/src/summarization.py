from config.settings import client

MODEL_NAME_DSLLAMA70B = "deepseek-r1-distill-llama-70b"
MODEL_NAME_DSQWEN32B = "deepseek-r1-distill-qwen-32b"
MODEL_NAME_QWEN32B = "qwen-2.5-32b"

TOKEN_LIMIT = 4096

def summarize_chunk(chunk):
    """Summarizes a single chunk using Groq LLM."""
    response = client.chat.completions.create(
        model=MODEL_NAME_DSLLAMA70B,
        messages=[
            {"role": "system", "content": "Summarize this text concisely."},
            {"role": "user", "content": chunk}
        ],
        temperature=0.6,
        max_completion_tokens=TOKEN_LIMIT,
        top_p=0.95
    )
    return response.choices[0].message.content

def summarize_chunks(chunks):
    """Summarizes each chunk in parallel."""
    return [summarize_chunk(chunk) for chunk in chunks]

def recursive_summarization(summaries):
    """Recursively summarizes multiple summaries until the final summary fits the LLM token limit."""
    while len(" ".join(summaries)) > 4000:  # Adjust to fit within LLM constraints
        summaries = summarize_chunks([" ".join(summaries[i:i+2]) for i in range(0, len(summaries), 2)])
    
    return " ".join(summaries)
