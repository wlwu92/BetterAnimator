from PIL import Image
import torch
import gc

from diffusers import (
    FluxPipeline,
    FluxFillPipeline,
    AutoencoderKL,
    FluxTransformer2DModel,
)
from diffusers.image_processor import VaeImageProcessor

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

flux_pipeline_info_map = {
    "flux": "models/FLUX/FLUX.1-dev",
    "flux_fill": "models/FLUX/FLUX.1-Fill-dev",
}


class MultiDevicePipelineBase:
    def __init__(self, pipeline_type: str = "flux", lora_name: str = ""):
        self._vae = None
        self._transformer = None
        self.lora_name = lora_name
        self.model_name = flux_pipeline_info_map[pipeline_type]

    def vae(self, device: str = "cpu"):
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            )
        return self._vae.to(device)

    def transformer(self):
        if self._transformer is None:
            self._transformer = FluxTransformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            if self.lora_name:
                self._transformer.load_lora_weights(self.lora_name)
        return self._transformer

    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def _decode_latents(
        self,
        latents: torch.Tensor,
        width: int,
        height: int,
    ) -> Image.Image:
        vae = self.vae("cuda")
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        with torch.no_grad():
            latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")[0]
        return image


class MultiDeviceFluxPipeline(MultiDevicePipelineBase):
    def __init__(self, lora_name: str = ""):
        super().__init__(pipeline_type="flux", lora_name=lora_name)

    def __call__(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        generator: torch.Generator = None,
    ) -> Image.Image:
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(
            prompt,
            max_sequence_length,
        )
        latents = self._denoise(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return self._decode_latents(latents, width, height)

    def _encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pipeline = FluxPipeline.from_pretrained(
            self.model_name,
            transformer=None,
            vae=None,
            device_map="balanced",
            max_memory={0: "16GB", 1: "16GB"},
            torch_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=max_sequence_length,
            )
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.tokenizer
        del pipeline.tokenizer_2
        del pipeline
        flush()
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _denoise(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        pipeline = FluxPipeline.from_pretrained(
            self.model_name,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            transformer=self.transformer(),
            torch_dtype=torch.bfloat16,
        )
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="latent",
            generator=generator,
        ).images
        del self._transformer
        del pipeline
        flush()
        return latents 

class MultiDeviceFluxFillPipeline(MultiDevicePipelineBase):
    def __init__(self):
        super().__init__(pipeline_type="flux_fill")

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        generator: torch.Generator = None,
    ) -> Image.Image:
        prompt_embeds, pooled_prompt_embeds, text_ids, masked_image_latents = \
            self._encode_prompt_and_prepare_mask_latents(
                prompt,
                image,
                mask_image,
                max_sequence_length,
                generator=generator,
            )
        latents = self._denoise(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            masked_image_latents=masked_image_latents,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return self._decode_latents(latents, width, height)

    def _encode_prompt_and_prepare_mask_latents(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        max_sequence_length: int = 512,
        generator: torch.Generator = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pipeline = FluxFillPipeline.from_pretrained(
            self.model_name,
            transformer=None,
            vae=self.vae("cpu"),
            device_map="balanced",
            max_memory={0: "16GB", 1: "16GB"},
            torch_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=max_sequence_length,
            )
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.tokenizer
        del pipeline.tokenizer_2
        flush()

        pipeline.vae.to("cuda")
        with torch.no_grad():
            width, height = image.size
            image = pipeline.image_processor.preprocess(image, height=height, width=width)
            mask_image = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)
            masked_image = image * (1 - mask_image)
            masked_image = masked_image.to(device="cuda:0", dtype=prompt_embeds.dtype)
            height, width = image.shape[-2:]
            num_channels_latents = pipeline.vae.config.latent_channels
            mask, masked_image_latents = pipeline.prepare_mask_latents(
                mask_image,
                masked_image,
                batch_size=1,
                num_channels_latents=num_channels_latents,
                num_images_per_prompt=1,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
                generator=generator,
            )
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
            masked_image_latents = masked_image_latents.to("cpu")
        self._vae = self._vae.to("cpu")
        flush()
        return prompt_embeds, pooled_prompt_embeds, text_ids, masked_image_latents
        
    def _denoise(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        masked_image_latents: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        pipeline = FluxFillPipeline.from_pretrained(
            self.model_name,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=self.vae("cpu"),
            transformer=self.transformer(),
            torch_dtype=torch.bfloat16,
        )
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="latent",
            masked_image_latents=masked_image_latents,
            generator=generator,
        ).images
        del self._transformer
        del pipeline
        flush()
        return latents

def flux_pipe(lora_name: str = "", enable_multi_gpu: bool = False) -> FluxPipeline:
    if enable_multi_gpu:
        return MultiDeviceFluxPipeline(lora_name=lora_name)

    pipe = FluxPipeline.from_pretrained(
        "models/FLUX/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    if lora_name:
        pipe.load_lora_weights(lora_name)
    pipe.enable_model_cpu_offload()
    return pipe


def flux_fill_pipe(enable_multi_gpu: bool = False) -> FluxFillPipeline:
    if enable_multi_gpu:
        return MultiDeviceFluxFillPipeline()

    pipe = FluxFillPipeline.from_pretrained(
        "models/FLUX/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe