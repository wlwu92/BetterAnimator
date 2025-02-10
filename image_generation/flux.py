from PIL import Image
import torch

from diffusers import FluxFillPipeline
from diffusers.utils import load_image

# Change to diffusers pipeline
from diffsynth import (
    ModelManager,
    FluxImagePipeline,
    ControlNetConfigUnit,
    download_models,
    download_customized_models
)

CONTROLNET_IDS_MODEL_PATH = {
    "tile": [
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors"
    ]
}

def flux_pipe(
    lora_name: str = "",
    controlnets_id_scale: list[(str, float)] = [],
    quantize: bool = False) -> FluxImagePipeline:
    model_id_list = ["FLUX.1-dev"]
    controlnet_config_units = []
    if controlnets_id_scale:
        for controlnet_id, scale in controlnets_id_scale:
            if controlnet_id not in CONTROLNET_IDS_MODEL_PATH:
                raise ValueError(f"Controlnet ID {controlnet_id}, supported controlnet ids: {list(CONTROLNET_IDS_MODEL_PATH.keys())}")
            model_id, model_path = CONTROLNET_IDS_MODEL_PATH[controlnet_id]
            if model_id not in model_id_list:
                model_id_list.append(model_id)
            controlnet_config_units.append(
                ControlNetConfigUnit(
                    processor_id=controlnet_id,
                    model_path=model_path,
                    scale=scale))
    download_models(model_id_list)
    if quantize:
        model_manager = ModelManager(
            device="cpu",
            torch_dtype=torch.bfloat16
        )
        model_manager.load_models([
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
            ])
        model_manager.load_models(
            ["models/FLUX/FLUX.1-dev/flux1-dev.safetensors"],
            torch_dtype=torch.float8_e4m3fn 
        )
        for controlnet_config_unit in controlnet_config_units:
            model_manager.load_models([controlnet_config_unit.model_path],
                torch_dtype=torch.float8_e4m3fn)
        if lora_name:
            model_manager.load_lora(
                lora_name, lora_alpha=0.8
            )
        pipe = FluxImagePipeline.from_model_manager(
            model_manager, device="cuda", controlnet_config_units=controlnet_config_units)
        pipe.enable_cpu_offload()
        pipe.dit.quantize()
        for model in pipe.controlnet.models:
            model.quantize()
    else:
        model_manager = ModelManager(
            device="cuda",
            torch_dtype=torch.bfloat16,
            model_id_list=model_id_list
        ) 
        pipe = FluxImagePipeline.from_model_manager(
            model_manager, controlnet_config_units=controlnet_config_units)
        if lora_name:
            pipe.load_lora(lora_name)
    return pipe

def flux_fill_pipe():
    pipe = FluxFillPipeline.from_pretrained(
        "models/FLUX/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16
    )
    return pipe