from inference.llama_inference import LLaMAInference

# Load Model for Inference
model_path = "./fine-tuned-model"  # Change to your model path
base_model_id = "meta-llama/Llama-3.2-1B"

llama_infer = LLaMAInference(model_path, base_model_id)
llama_infer.load_model()

# Get User Input
while True:
    prompt = input("\nEnter a medical query (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    response = llama_infer.generate_response(prompt, max_length=256)
    
    print("\nMedical Chatbot Response:")
    print(response)
