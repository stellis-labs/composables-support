class DataConfig:
    def __init__(self):
        self.dataset_name = "mlabonne/FineTome-100k"
        self.split = "train"
        self.chat_template = "llama-3.1"
        self.response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"