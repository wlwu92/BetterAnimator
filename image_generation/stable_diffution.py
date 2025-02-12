import torch
import os
from typing import List

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from huggingface_hub import snapshot_download

from common.constant import HUGGINGFACE_CACHE_DIR, DEVICE

CONTROLNET_MODELS = {
    "tile": "models/ControlNet/control_v11f1e_sd15_tile.pth",
    "lineart": "models/ControlNet/control_v11p_sd15_lineart.pth",
}

def _get_controlnet_model(control_unit: str) -> ControlNetModel:
    assert control_unit in CONTROLNET_MODELS, f"Control unit {control_unit} not found"
    model_path = CONTROLNET_MODELS[control_unit]
    if not os.path.exists(model_path):
        snapshot_download(
            repo_id="lllyasviel/ControlNet-v1-1",
            local_dir=os.path.dirname(model_path),
            allow_patterns=os.path.basename(model_path),
        )
    model = ControlNetModel.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    )
    return model

def sd_text2img_pipe(model_path: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "mps":
        pipe.to(DEVICE)
    else:
        pipe.enable_model_cpu_offload()
    return pipe

def sd_img2img_pipe(model_path: str) -> StableDiffusionImg2ImgPipeline:
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "mps":
        pipe.to(DEVICE)
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def sd_controlnet_pipe(model_path: str, control_units: List[str]) -> StableDiffusionControlNetPipeline:
    controlnets = [
        _get_controlnet_model(unit)
        for unit in control_units
    ]
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        controlnet=controlnets,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "mps":
        pipe.to(DEVICE)
    else:
        pipe.enable_model_cpu_offload()
    return pipe
