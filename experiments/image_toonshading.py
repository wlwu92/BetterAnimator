import argparse
import os
import random

from PIL import Image
import torch

from diffsynth import ModelManager, SDImagePipeline, ControlNetConfigUnit, download_models

def save_generated_image(image_file, gen_image, prompt, seed):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]
    gen_dir = os.path.join(os.path.dirname(image_file), f"{image_name}_gen")
    os.makedirs(gen_dir, exist_ok=True)
    
    existing_files = os.listdir(gen_dir)
    
    max_id = 0
    for file in existing_files:
        if file.endswith(".png"):
            try:
                file_id = int(os.path.splitext(file)[0].split('_')[-1])
                if file_id > max_id:
                    max_id = file_id
            except ValueError:
                continue
    gen_id = max_id + 1
    gen_image_file = os.path.join(gen_dir, f"gen_{gen_id}.png")
    gen_image.save(gen_image_file)
    prompt_file = os.path.join(gen_dir, f"prompt_{gen_id}_seed_{seed}.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

def image_toon_shading(image_file, prompt, seed):
    download_models(["AingDiffusion_v12", "ControlNet_v11p_sd15_lineart", "ControlNet_v11f1e_sd15_tile"])
    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/stable_diffusion/aingdiffusion_v12.safetensors",
                                 "models/ControlNet/control_v11f1e_sd15_tile.pth",
                                 "models/ControlNet/control_v11p_sd15_lineart.pth"
                             ])
    pipe = SDImagePipeline.from_model_manager(
        model_manager,
        [
            ControlNetConfigUnit(
                processor_id="tile",
                model_path=rf"models/ControlNet/control_v11f1e_sd15_tile.pth",
                scale=0.5
            ),
            ControlNetConfigUnit(
                processor_id="lineart",
                model_path=rf"models/ControlNet/control_v11p_sd15_lineart.pth",
                scale=0.5
            ),
        ]
    )
    image = Image.open(image_file)
    image = image.resize((image.width // 64 * 64, image.height // 64 * 64))
    return pipe(
        prompt=prompt,
        input_image=image,
        controlnet_image=image,
        cfg_scale=7.0,
        clip_skip=2,
        denoising_strength=1.0,
        height=image.height,
        width=image.width,
        num_inference_steps=10,
        seed=seed
    )

def main():
    parser = argparse.ArgumentParser(description="Process a video with a given prompt.")
    parser.add_argument('image_file', type=str, help='Path to the image file')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for the random number generator')
    
    args = parser.parse_args()
    
    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 1000000)
    
    prompt_file_path = os.path.join(os.path.dirname(args.image_file), 'prompt.txt')
    
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read().strip()
    
    gen_image = image_toon_shading(args.image_file, prompt, seed)
    save_generated_image(args.image_file, gen_image, prompt, seed)


if __name__ == "__main__":
    main()