import json
from PIL import Image
import random
import argparse

from common.constant import CHARACTER_DIR
from image_generation.stable_diffution import sd_controlnet_img2img_pipe_v2

def animate_image_random_seed(image_path, prompt, negative_prompt, output_path, times=1):
    pipe = sd_controlnet_img2img_pipe_v2(
        model_path="./models/stable_diffusion/aingdiffusion_v12.safetensors",
        control_units=["tile", "lineart"],
    )
    image = Image.open(image_path)
    width, height = image.size
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    input_image = image.resize((new_width, new_height))
    for i in range(times):
        seed = random.randint(0, 1000000)
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            controlnet_image=input_image,
            num_inference_steps=10,
            cfg_scale=7.0,
            clip_skip=2,
            denoising_strength=1.0,
            height=new_height,
            width=new_width,
            seed=seed
        )
        output_image = output_image.resize((width, height))
        output_path_i = output_path.parent / f"{output_path.stem}_{i:03d}_seed_{seed}.png"
        output_image.save(output_path_i)

def animate_image(image_path, prompt, negative_prompt, output_path, seed=0):
    pipe = sd_controlnet_img2img_pipe_v2(
        model_path="./models/stable_diffusion/aingdiffusion_v12.safetensors",
        control_units=["tile", "lineart"],
    )
    image = Image.open(image_path)
    width, height = image.size
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    input_image = image.resize((new_width, new_height))
    output_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        controlnet_image=input_image,
        num_inference_steps=20,
        cfg_scale=7.0,
        clip_skip=2,
        denoising_strength=1.0,
        height=new_height,
        width=new_width,
        seed=seed
    )
    output_image = output_image.resize((width, height))
    output_image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Animate an image')
    parser.add_argument('--character_id', type=str, required=True, help='Character id')
    parser.add_argument('--times', type=int, default=1, help='Number of times to random seed')
    parser.add_argument('--seed', type=int, default=-2, help='Seed')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    character_dir = CHARACTER_DIR / args.character_id
    assert character_dir.exists(), f"Character directory {character_dir} does not exist"
    character_image_path = character_dir / "character.png"
    assert character_image_path.exists(), f"Character image {character_image_path} does not exist"
    prompt_file = character_dir / "prompt.json"
    assert prompt_file.exists(), f"Prompt file {prompt_file} does not exist"
    seed = args.seed
    with open(prompt_file, "r") as f:
        prompt_info = json.load(f)
        prompt = prompt_info["prompt"]
        negative_prompt = prompt_info.get("negative_prompt", "verybadimagenegative_v1.3, embroidery, printed patterns, graphic design elements")
        if seed == -2:
            seed = prompt_info.get("seed", -1)
    if seed != -1:
        assert args.times == 1, "Times must be 1 when seed is -1"
    output_path = character_dir / "character_anime.png"
    if args.debug:
        output_dir = character_dir / "anime"
        output_dir.mkdir(parents=True, exist_ok=True)
        import datetime
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"character_anime_{time_str}.png"
        prompt_save_path = output_path.with_suffix(".json")
        with open(prompt_save_path, "w") as f:
            json.dump(
                {"prompt": prompt, "seed": seed, "negative_prompt": negative_prompt},
                f,
                indent=4)
    if seed != -1:
        animate_image(character_image_path, prompt, negative_prompt, output_path, seed)
    else:
        animate_image_random_seed(character_image_path, prompt, negative_prompt, output_path, times=args.times)

if __name__ == "__main__":
    main()
