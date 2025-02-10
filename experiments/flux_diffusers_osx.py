import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
    torch_dtype=torch.float16,
).to(device)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.float16,
).to(device)

# pipe.enable_model_cpu_offload()
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
# Generate failed with black image
image.save("flux-gguf.png")

