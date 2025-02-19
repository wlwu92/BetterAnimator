import os
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch

from image_generation.flux import flux_fill_pipe

reference_scales_params = {
    "x1": [25, 5],
    "x2": [35, 10],
    "x3": [50, 15]
}

MULTI_DEVICE_INFERENCE = os.environ.get("MULTI_DEVICE_INFERENCE", "0") == "1"

PROMPT = "simple color background"
outpaint_pipe = None
def load_outpaint_pipe():
    global outpaint_pipe
    outpaint_pipe = flux_fill_pipe(
        lora_name="models/FLUX/F.1_FitnessTrainer_lora_v1.0.safetensors",
        enable_multi_gpu=MULTI_DEVICE_INFERENCE,
        use_quantization=False,
    )

def image_detect_one_object(image_path: str) -> Results:
    model = YOLO("models/yolo11n-pose.pt")
    results = model.predict(image_path, verbose=True)
    # TODO(wanglong): Add a method to select the maximum confidence
    return results[0].to("cpu")

def safe_crop_and_resize(image: Image.Image, box: tuple, target_size: tuple) -> Image.Image:
    """Safely crop and resize the image, handling cases where it exceeds the boundaries
    
    Args:
        image: PIL Image object
        box: Cropping box (left, top, right, bottom)
        target_size: Target size (width, height)
    
    Returns:
        The cropped and resized image
    """
    # Get the original image size
    img_width, img_height = image.size
    left, top, right, bottom = box
    
    # Calculate the boundaries that need to be padded
    pad_left = abs(min(0, left))
    pad_top = abs(min(0, top))
    pad_right = max(0, right - img_width)
    pad_bottom = max(0, bottom - img_height)
    
    # If padding is needed, create a new canvas
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        # Use cv2.copyMakeBorder to add padding with edge color
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        padded_image = cv2.copyMakeBorder(
            cv_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE
        )
        image = Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
        # Adjust the cropping box coordinates
        box = (
            left + pad_left,
            top + pad_top,
            right + pad_left,
            bottom + pad_top
        )
    
    # Now it is safe to crop
    cropped = image.crop(box)
    
    # Adjust to the target size
    resized = cropped.resize(target_size, Image.Resampling.BICUBIC)
    
    return resized

def transform_bbox(target_bbox: np.ndarray, target_width: int, target_height: int, bbox: np.ndarray) -> np.ndarray:
    height = target_bbox[3] - target_bbox[1]
    width = target_bbox[2] - target_bbox[0]

    bbox[0] = (bbox[0] - target_bbox[0]) / width * target_width
    bbox[1] = (bbox[1] - target_bbox[1]) / height * target_height
    bbox[2] = (bbox[2] - target_bbox[0]) / width * target_width
    bbox[3] = (bbox[3] - target_bbox[1]) / height * target_height
    return bbox

def outpaint_image(image: Image.Image, image_bbox: np.ndarray, object_bbox: np.ndarray, num_inference_steps=5):
    def _fit_to(value, min_value, max_value, div=8):
        if max_value < min_value:
            min_value, max_value = max_value, min_value
        value = (value // div) * div
        if min_value < value < max_value:
            return value
        value += div
        if min_value < value < max_value:
            return value
        return None 
        
    if image_bbox[0] > 0 or image_bbox[1] > 0 or image_bbox[2] < image.size[0] or image_bbox[3] < image.size[1]:
        masked_bbox = (image_bbox + object_bbox) / 2
        masked_bbox = np.round(masked_bbox).astype(int)
        min_gap_x, min_gap_y = 36, 64
        if masked_bbox[0] - image_bbox[0] > min_gap_x:
            masked_bbox[0] = image_bbox[0] + min_gap_x
        if masked_bbox[1] - image_bbox[1] > min_gap_y:
            masked_bbox[1] = image_bbox[1] + min_gap_y
        if image_bbox[2] - masked_bbox[2] > min_gap_x:
            masked_bbox[2] = image_bbox[2] - min_gap_x
        if image_bbox[3] - masked_bbox[3] > min_gap_y:
            masked_bbox[3] = image_bbox[3] - min_gap_y
        for i in range(4):
            for div in [16, 8, 4]:
                value = _fit_to(masked_bbox[i], image_bbox[i], object_bbox[i], div)
                if value is not None:
                    masked_bbox[i] = value
                    break

        mask = Image.new("L", image.size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([
            int(masked_bbox[0]), int(masked_bbox[1]),
            int(masked_bbox[2]), int(masked_bbox[3])
        ], fill=0)
        
        if outpaint_pipe is None:
            load_outpaint_pipe()
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        outpainted_image = outpaint_pipe(
            prompt=PROMPT,
            image=image,
            mask_image=mask.convert("RGB"),
            height=image.height,
            width=image.width,
            guidance_scale=3.5,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        return outpainted_image, mask

    return image, None


def get_object_bbox(result):
    bbox = result.boxes.xyxy[0].numpy()
    center_x = result.keypoints[0].xy[0, :, 0].numpy().mean()
    width = max(abs(bbox[2] - center_x), abs(center_x - bbox[0])) * 2
    bbox[0] = center_x - width / 2
    bbox[2] = center_x + width / 2
    return bbox


def generate_character_scales(
    character_dir: str,
    update_scale: str = None,
    num_inference_steps: int = 2) -> None:
    if update_scale is None or update_scale == "x1":
        base_image_path = os.path.join(character_dir, "character.png")
    else:
        base_scale = int(update_scale.strip("x")) - 1
        base_scale = f'x{base_scale}'
        assert base_scale in reference_scales_params, f"Invalid base scale: {base_scale}"
        base_image_path = os.path.join(character_dir, f"character_{base_scale}.png")
    print(f"Detecting object in {base_image_path}")
    result = image_detect_one_object(base_image_path)
    object_bbox = get_object_bbox(result)
    image = Image.open(base_image_path)

    for scale, params in reference_scales_params.items():
        if update_scale is not None and scale != update_scale:
            continue
        center_x = (object_bbox[0] + object_bbox[2]) / 2
        height = object_bbox[3] - object_bbox[1]
        scaled_bbox = object_bbox.copy()
        scaled_bbox[1] -= height * params[0] / 100
        scaled_bbox[3] += height * params[1] / 100
        height = scaled_bbox[3] - scaled_bbox[1]
        width = height * 9 / 16
        scaled_bbox[0] = center_x - width / 2
        scaled_bbox[2] = center_x + width / 2
        scaled_bbox = np.round(scaled_bbox).astype(int)

        new_width, new_height = 864, 1536

        scaled_image = safe_crop_and_resize(image, scaled_bbox, (new_width, new_height))
        image_bbox = transform_bbox(scaled_bbox, new_width, new_height, np.array([0, 0, image.size[0], image.size[1]]))
        object_bbox = transform_bbox(scaled_bbox, new_width, new_height, object_bbox)
        outpainted_image, _ = outpaint_image(
            scaled_image,
            image_bbox,
            object_bbox,
            num_inference_steps=num_inference_steps
        )
        outpainted_image.save(f"{character_dir}/character_{scale}.png")
        image = outpainted_image