from inference.llama_inference import LLaMAInference
from inference.qwen_inference import QwenInference

# Select Model
model_choice = input("Choose model: 'llama' or 'qwen': ").strip().lower()

if model_choice == "llama":
    model_path = "./fine-tuned-llama"
    base_model_id = "meta-llama/Llama-3.2-1B"
    inference_model = LLaMAInference(model_path, base_model_id)
elif model_choice == "qwen":
    model_path = "./fine-tuned-qwen"
    base_model_id = "Qwen/Qwen1.5-1.8B"
    inference_model = QwenInference(model_path, base_model_id)
else:
    raise ValueError("Invalid model selection!")

inference_model.load_model()

# Run inference
while True:
    prompt = input("\nEnter a query (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    response = inference_model.generate_response(prompt)
    print("\nModel Response:")
    print(response)
