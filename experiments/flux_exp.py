from image_generation.flux import flux_pipe
import torch
from PIL import Image
import numpy as np
import os

prompt = "Yoga pants, A young woman's energetic middle pose. The photo is a side view of her full body,in a running pose,with her right leg bent forward and her left leg extended back as if she were stepping. Her facial expression is firm and focused. She had fair skin and her hair was pulled back into a high ponytail,adding a touch of sporty elegance. She wore a white crop top that accentuated her slender torso and a pair of high-waisted black leggings that framed her toned figure and accentuated her toned legs and hips. Her shoes consist of white sneakers with a minimalist design that emphasizes function over fashion. The background is a simple off-white color that acts as a neutral background,ensuring that the focus remains on her movement form. The overall image style is clean and modern,with a focus on fitness and an active lifestyle. A balanced composition with a clear emphasis on sport and fitness,suitable for health and fitness related media or advertising. ,"
negative_prompt = "ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),"
output_dir = "outputs/flux_exp"
os.makedirs(output_dir, exist_ok=True)

def flux_generate_image():
    pipe = flux_pipe(quantize=True)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=640, width=384,
        seed=0
    )
    image.save(os.path.join(output_dir, "flux_image.jpg"))

def flux_lora_generate_image():
    pipe = flux_pipe(
        lora_name="models/FLUX/F.1_FitnessTrainer_lora_v1.0.safetensors",
        quantize=True)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=640, width=384,
        seed=0
    )
    image.save(os.path.join(output_dir, "flux_lora_image.jpg"))

def flux_controlnet_generate_image():
    pipe = flux_pipe(
        controlnets_id_scale=[
            ("tile", 0.6)
        ],
        quantize=True
    )
    controlnet_image = Image.open("data/flux_controlnet_blur.png")
    controlnet_image = controlnet_image.resize((384, 640))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_image=controlnet_image,
        height=640, width=384,
        seed=0
    )
    image.save(os.path.join(output_dir, "flux_controlnet_image.jpg"))

def flux_inpaint_generate_image():
    image = Image.open("data/flux_controlnet_blur.png")
    image = image.resize((384, 640))

    mask = np.zeros((640, 384, 3), dtype=np.uint8)
    mask[180:230, 50:100] = 255
    mask[120:160, 300:350] = 255
    mask = Image.fromarray(mask)
    mask.save(os.path.join(output_dir, "mask_9.jpg"))
    # draw mask on image
    mask_paste = mask.convert("L")
    image_paste = image.copy()
    image_paste.paste(mask_paste, (0, 0), mask_paste)
    image_paste.save(os.path.join(output_dir, "mask_on_image.jpg"))

    pipe = flux_pipe(
        lora_name="models/FLUX/F.1_FitnessTrainer_lora_v1.0.safetensors",
        quantize=True
    )
    image_2 = pipe(
        prompt=prompt,
        local_prompts=["Masterpiece, High Definition, Real Person Portrait, 5 Fingers, Girl's Hand",],
        masks=[mask,], mask_scales=[2.0],
        denoising_strength=0.4,
        num_inference_steps=20,
        input_image=image,
        height=640, width=384,
        seed=-1
    )
    image_2.save(os.path.join(output_dir, "flux_inpaint_image_scale_2_0.3_30_lora.jpg"))

flux_generate_image()
flux_lora_generate_image()
flux_controlnet_generate_image()
flux_inpaint_generate_image()