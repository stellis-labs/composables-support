from .llama import LLaMAModel
from .qwen import QwenModel

def get_model(model_name, model_id):
    """Returns the appropriate model class based on user input."""
    if model_name.lower() == "llama":
        return LLaMAModel(model_id)
    elif model_name.lower() == "qwen":
        return QwenModel(model_id)
    else:
        raise ValueError("Unsupported model type! Choose 'llama' or 'qwen'.")
