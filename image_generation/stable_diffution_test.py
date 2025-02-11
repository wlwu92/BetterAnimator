import unittest
from PIL import Image

from image_generation.stable_diffution import (
    sd_text2img_pipe,
    sd_img2img_pipe,
    sd_controlnet_pipe,
)

class TestImageGeneration(unittest.TestCase):
    def test_image_generation(self):
        text2img_pipe = sd_text2img_pipe("./models/stable_diffusion/aingdiffusion_v12.safetensors")
        image = text2img_pipe("a photo of an astronaut riding a horse on mars", num_inference_steps=20).images[0]
        image.save("outputs/test.png")

        img2img_pipe = sd_img2img_pipe("./models/stable_diffusion/aingdiffusion_v12.safetensors")
        image = img2img_pipe(
            "a photo of an astronaut riding a horse on mars",
            image=Image.open("outputs/test.png"),
            strength=0.8,
            num_inference_steps=20,
            guidance_scale=7.5,
            negative_prompt="",
            num_images_per_prompt=1,
            eta=0.0,
        ).images[0]
        image.save("outputs/test_img2img.png")
        
    def test_controlnet(self):
        input_image = Image.open("data/example_reference/ref.png")
        controlnet_pipe = sd_controlnet_pipe(
            "./models/stable_diffusion/aingdiffusion_v12.safetensors",
            ["tile", "lineart"],
        )
        image = controlnet_pipe(
            "A girl in a red dress",
            image=[input_image, input_image],
            controlnet_conditioning_scale=[0.5, 0.5],
            num_inference_steps=20,
            guidance_scale=7.5,
            negative_prompt="",
            num_images_per_prompt=1,
            eta=0.0,
        ).images[0]
        image.save("outputs/test_controlnet.png")
if __name__ == '__main__':
    unittest.main()