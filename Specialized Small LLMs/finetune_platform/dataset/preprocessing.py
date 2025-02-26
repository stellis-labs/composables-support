def preprocess_text(examples):
    """Formats dataset into prompt-response format"""
    instruction_texts = [" ".join(inst) if isinstance(inst, list) else inst for inst in examples["instruction"]]
    input_texts = [" ".join(inp) if isinstance(inp, list) else inp for inp in examples["input"]]
    output_texts = [" ".join(out) if isinstance(out, list) else out for out in examples["output"]]

    formatted_texts = []
    for instr, inp, out in zip(instruction_texts, input_texts, output_texts):
        prompt = f"<|user|>\n{instr.strip()}"
        if inp.strip():
            prompt += f"\n{inp.strip()}"
        prompt += f"\n\n<|assistant|>\n{out.strip()}"
        formatted_texts.append(prompt)

    return formatted_texts
