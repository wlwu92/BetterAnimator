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
    "x2": [40, 10],
    "x3": [55, 15]
}

PROMPT = "Consistant background, a woman in standing in a simple color background. The background is a simple off-white color that acts as a neutral canvas,ensuring that the focus remains on her posture and presence. The overall image style is clean and modern,with a focus on fitness and an active lifestyle"
outpaint_pipe = None
def load_outpaint_pipe():
    global outpaint_pipe
    outpaint_pipe = flux_fill_pipe()

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
        # Create a new canvas (fill with black or other color)
        new_width = img_width + pad_left + pad_right
        new_height = img_height + pad_top + pad_bottom
        new_image = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
        
        # Paste the original image to the correct position on the new canvas
        new_image.paste(image, (pad_left, pad_top))
        
        # Adjust the cropping box coordinates
        box = (
            left + pad_left,
            top + pad_top,
            right + pad_left,
            bottom + pad_top
        )
        image = new_image
    
    # Now it is safe to crop
    cropped = image.crop(box)
    
    # Adjust to the target size
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized

def transform_bbox(target_bbox: np.ndarray, target_width: int, target_height: int, bbox: np.ndarray) -> np.ndarray:
    height = target_bbox[3] - target_bbox[1]
    width = target_bbox[2] - target_bbox[0]

    bbox[0] = (bbox[0] - target_bbox[0]) / width * target_width
    bbox[1] = (bbox[1] - target_bbox[1]) / height * target_height
    bbox[2] = (bbox[2] - target_bbox[0]) / width * target_width
    bbox[3] = (bbox[3] - target_bbox[1]) / height * target_height
    return bbox

def outpaint_image(image: Image.Image, image_bbox: np.ndarray, object_bbox: np.ndarray):
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
            input_image=image,
            mask_image=mask.convert("RGB"),
            height=image.height,
            width=image.width,
            guidance_scale=30,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
            seed=0
        )
        return outpainted_image, mask

    return image, None


def get_object_bbox(result):
    bbox = result.boxes.xyxy[0].numpy()
    center_x = result.keypoints[0].xy[0, :, 0].numpy().mean()
    width = max(abs(bbox[2] - center_x), abs(center_x - bbox[0])) * 2
    bbox[0] = center_x - width / 2
    bbox[2] = center_x + width / 2
    return bbox


def generate_character_scales(character_dir: str) -> None:
    image = Image.open(os.path.join(character_dir, "character.png"))
    result = image_detect_one_object(os.path.join(character_dir, "character.png"))
    object_bbox = get_object_bbox(result)

    boxes_list = []
    for scale, params in reference_scales_params.items():
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

        new_width, new_height = 720, 1280
        scaled_image = safe_crop_and_resize(image, scaled_bbox, (new_width, new_height))
        image_bbox = transform_bbox(scaled_bbox, new_width, new_height, np.array([0, 0, image.size[0], image.size[1]]))
        object_bbox = transform_bbox(scaled_bbox, new_width, new_height, object_bbox)
        step = 0
        while step < 5:
            outpainted_image, mask = outpaint_image(scaled_image, image_bbox, object_bbox)
            if mask is None:
                break
            # Calculate the area of the black region in the mask
            mask_array = np.array(mask)
            total_area = np.sum(mask_array > 0)
            outpainted_image_array = np.array(outpainted_image)
            black_area = np.sum(outpainted_image_array[mask_array > 0] == 0)
            black_area_ratio = black_area / total_area
            
            # If the area of the black region is greater than 20%, recalculate
            if black_area_ratio > 0.2:
                outpainted_image.save(f"{character_dir}/scale_{scale}_step_{step}.png")
                step += 1
                continue
            break
        outpainted_image.save(f"{character_dir}/character_{scale}.png")
        # if mask is not None:
        #     masked_scaled_image = scaled_image.copy()
        #     # Multiply the red channel of the mask area by 2
        #     image_array = np.array(masked_scaled_image)
        #     mask_array = np.array(mask)
        #     image_array[mask_array == 0, 0] = np.clip(image_array[mask_array == 0, 0] + 50, 0, 255)
        #     masked_scaled_image = Image.fromarray(image_array)
        #     masked_scaled_image.save(f"/tmp/masked_scaled_{scale}.png")
        image = outpainted_image