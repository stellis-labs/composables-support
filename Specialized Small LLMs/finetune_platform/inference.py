import json
from inference.llama_inference import LLaMAInference
from inference.qwen_inference import QwenInference
from inference.llama_unsloth_inference import LLaMAUnslothInference 
from inference.qwen_unsloth_inference import QwenUnslothInference

# Load default inference parameters from config file
config_path = "config/inference_config.json"
with open(config_path, "r") as file:
    config = json.load(file)

# Selection Menu
print("\nChoose a model for inference:")
print("1. LLaMA (Hugging Face)")
print("2. LLaMA (Unsloth)")
print("3. Qwen (Hugging Face)")
print("4. Qwen (Unsloth)")

# Get User Selection
choice = input("\nEnter your choice (1-4): ").strip()

if choice == "1":
    model_path = "./fine-tuned-llama"
    base_model_id = "meta-llama/Llama-3.2-1B"
    inference_model = LLaMAInference(model_path, base_model_id)
elif choice == "2":
    model_path = "./fine-tuned-llama-unsloth"
    base_model_id = "meta-llama/Llama-3.2-1B"
    inference_model = LLaMAUnslothInference(model_path, base_model_id)
elif choice == "3":
    model_path = "./fine-tuned-qwen"
    base_model_id = "Qwen/Qwen1.5-1.8B"
    inference_model = QwenInference(model_path, base_model_id)
elif choice == "4":
    model_path = "./fine-tuned-qwen-unsloth"
    base_model_id = "Qwen/Qwen1.5-1.8B"
    inference_model = QwenUnslothInference(model_path, base_model_id)
else:
    raise ValueError("Invalid selection! Please choose a valid option.")

# Load Model
print("\nLoading model for inference...")
inference_model.load_model()

# Ask user if they want to override parameters
override = input("\nDo you want to override inference parameters? (Y/N): ").strip().lower() == "y"

if override:
    config["max_length"] = int(input("Enter max response length (default 256): ") or config["max_length"])
    config["temperature"] = float(input("Enter temperature (default 0.7): ") or config["temperature"])
    config["top_p"] = float(input("Enter top_p (default 0.9): ") or config["top_p"])
    config["do_sample"] = input("Enable sampling? (Y/N, default yes): ").strip().lower() == "y"

# Run Inference Loop
while True:
    prompt = input("\nEnter a query (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    response = inference_model.generate_response(
        prompt,
        max_length=config["max_length"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        do_sample=config["do_sample"]
    )

    print("\nModel Response:")
    print(response)
