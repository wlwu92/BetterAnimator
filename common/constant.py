
HUGGINGFACE_CACHE_DIR = "./models/huggingface"

DEVICE = None
if DEVICE is None:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        raise ValueError("No device available")