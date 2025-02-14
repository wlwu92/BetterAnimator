from PIL import Image
import random
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
        output_path_i = output_path.replace(".png", f"_{i:03d}_seed_{seed}.png")
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