from .llama import LLaMAModel
from .qwen import QwenModel

def get_model(model_name, model_id, use_unsloth=False):
    """Returns the appropriate model class based on user input."""
    if model_name.lower() == "llama":
        return LLaMAModel(model_id, use_unsloth=use_unsloth)
    elif model_name.lower() == "qwen":
        return QwenModel(model_id, use_unsloth=use_unsloth)
    else:
        raise ValueError("Unsupported model type! Choose 'llama' or 'qwen'.")
