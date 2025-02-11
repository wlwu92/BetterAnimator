import unittest

from PIL import Image, ImageDraw

from image_generation.flux import flux_pipe, flux_fill_pipe

class TestFluxPipeline(unittest.TestCase):
    def test_flux_pipeline(self):
        pipe = flux_pipe()
        image = pipe("a photo of a dog with cat-like look",
             height=720,
             width=1280,
             num_inference_steps=2,
             guidance_scale=3.5,
             max_sequence_length=512,
        )
        image.save("flux_pipeline.png")


    def test_flux_fill_pipe(self):
        pipe = flux_fill_pipe()
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
        )
        image.save("flux_fill_pipe.png")
if __name__ == '__main__':
    unittest.main()