from pathlib import Path

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

WORKSPACE_DIR = Path("./data/workspace")
CHARACTER_DIR = WORKSPACE_DIR / "characters"
TASK_DIR = WORKSPACE_DIR / "gens"
VIDEO_DIR = WORKSPACE_DIR / "videos"
MANUAL_DIR = WORKSPACE_DIR / "manual"
