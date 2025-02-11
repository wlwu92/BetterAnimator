import unittest

from PIL import Image, ImageDraw

from image_generation.flux import flux_pipe, flux_fill_pipe, flux_img2img_pipe

class TestFluxPipeline(unittest.TestCase):
    def test_flux_pipeline(self):
        pipe = flux_pipe(enable_multi_gpu=True)
        image = pipe("a photo of a dog with cat-like look",
             height=720,
             width=1280,
             num_inference_steps=2,
             guidance_scale=3.5,
             max_sequence_length=512,
        ).images[0]
        image.save("flux_pipeline.png")

    def test_flux_img2img_pipeline(self):
        pipe = flux_img2img_pipe(enable_multi_gpu=True)
        image = Image.open("data/example_reference/ref.png")
        image = pipe("a beautiful girl",
             image=image,
             height=1280,
             width=720,
             num_inference_steps=2,
             guidance_scale=3.5,
             max_sequence_length=512,
        ).images[0]
        image.save("flux_img2img_pipeline.png")

    def test_flux_fill_pipeline(self):
        pipe = flux_fill_pipe(enable_multi_gpu=True)
        image = Image.open("data/example_reference/ref.png")
        mask = Image.new("L", image.size, 255)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([
            (100, 100),
            (200, 200),
        ], fill=0)
        import torch
        image = pipe(
            prompt="A beautiful girl",
            image=image,
            mask_image=mask,
            height=image.size[1],
            width=image.size[0],
            guidance_scale=30,
            num_inference_steps=2,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save("flux_fill_pipe.png")
    
    
if __name__ == '__main__':
    unittest.main()