import unittest
from PIL import Image
import torch

from image_generation.stable_diffution import (
    sd_text2img_pipe,
    sd_img2img_pipe,
    sd_controlnet_pipe,
    sd_controlnet_img2img_pipe,
    sd_controlnet_img2img_pipe_v2,
)

class TestImageGeneration(unittest.TestCase):
    def test_image_generation(self):
        text2img_pipe = sd_text2img_pipe("./models/stable_diffusion/aingdiffusion_v12.safetensors")
        image = text2img_pipe("a photo of an astronaut riding a horse on mars", num_inference_steps=20).images[0]
        image.save("outputs/sd_1.5_text2img.png")

        img2img_pipe = sd_img2img_pipe("./models/stable_diffusion/aingdiffusion_v12.safetensors")
        image = img2img_pipe(
            "a photo of an astronaut riding a horse on mars",
            image=Image.open("outputs/sd1.5_text2img.png"),
            strength=0.8,
            num_inference_steps=20,
            guidance_scale=7.5,
            negative_prompt="",
            num_images_per_prompt=1,
            eta=0.0,
        ).images[0]
        image.save("outputs/sd_1.5_img2img.png")
        
    def test_controlnet(self):
        input_image = Image.open("data/example_reference/ref.png")
        controlnet_pipe = sd_controlnet_pipe(
            "./models/stable_diffusion/aingdiffusion_v12.safetensors",
            ["tile", "lineart"],
        )
        image = controlnet_pipe(
            "best quality, perfect anime illustration, light, one girl, smile, solo",
            negative_prompt="verybadimagenegative_v1.3, embroidery, printed patterns, graphic design elements",
            image=[input_image, input_image],
            controlnet_conditioning_scale=[0.5, 0.5],
            num_inference_steps=10,
            strength=1.0,
            guidance_scale=7.0,
            clip_skip=2,
            num_images_per_prompt=1,
        ).images[0]
        image.save("outputs/sd_1.5_controlnet.png")

    def test_controlnet_img2img(self):
        input_image = Image.open("data/example_reference/ref.png")
        controlnet_pipe = sd_controlnet_img2img_pipe(
            "./models/stable_diffusion/aingdiffusion_v12.safetensors",
            ["tile", "lineart"],
            textual_inversion_path="./models/textual_inversion/verybadimagenegative_v1.3.pt",
        )
        image = controlnet_pipe(
            "best quality, perfect anime illustration, light, one girl, smile, solo",
            negative_prompt="verybadimagenegative_v1.3, embroidery, printed patterns, graphic design elements",
            image=input_image,
            control_image=[input_image, input_image],
            controlnet_conditioning_scale=[0.5, 0.5],
            guidance_scale=7.0,
            clip_skip=2,
            strength=1.0,
            num_inference_steps=10,
            generator=torch.Generator(device="cpu").manual_seed(42),
            height=1536,
            width=864,
        ).images[0]
        image.save("outputs/sd_1.5_controlnet_img2img.png")

    def test_controlnet_img2img_v2(self):
        input_image = Image.open("data/example_reference/ref.png")
        controlnet_pipe = sd_controlnet_img2img_pipe_v2(
            "./models/stable_diffusion/aingdiffusion_v12.safetensors",
            ["tile", "lineart"],
        )
        input_image = input_image.resize((input_image.width // 64 * 64, input_image.height // 64 * 64))
        image = controlnet_pipe(
            prompt="best quality, perfect anime illustration, light, one girl, smile, solo",
            input_image=input_image,
            controlnet_image=input_image,
            num_inference_steps=10,
            cfg_scale=7.0,
            clip_skip=2,
            denoising_strength=1.0,
            height=input_image.height,
            width=input_image.width,
        )
        image.save("outputs/sd_1.5_controlnet_img2img_v2.png")

if __name__ == '__main__':
    unittest.main()