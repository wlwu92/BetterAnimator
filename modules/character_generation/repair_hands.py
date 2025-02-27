import os

from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import ConvexHull
import numpy as np
import cv2

import torch

from image_generation.flux import flux_fill_pipe, flux_img2img_pipe

MULTI_DEVICE_INFERENCE = os.environ.get("MULTI_DEVICE_INFERENCE", "0") == "1"

deblur_pipe = None
def load_deblur_pipe():
    global deblur_pipe
    deblur_pipe = flux_img2img_pipe(enable_multi_gpu=MULTI_DEVICE_INFERENCE)

inpaint_pipe = None
def load_inpaint_pipe():
    global inpaint_pipe
    inpaint_pipe = flux_fill_pipe(
        lora_name="models/FLUX/F.1_FitnessTrainer_lora_v1.0.safetensors",
        enable_multi_gpu=MULTI_DEVICE_INFERENCE,
        use_quantization=False
    )

from pose_estimation.utils import (
    load_pose,
    draw_pose_on_image,
    get_pose_parts,
    to_openpose_format,
)

def get_keypoints_polygon_mask(keypoints_list: list[np.ndarray], image_size: tuple) -> Image.Image:
    """
    Get the polygon mask from the keypoints.
    """
    def _get_polygon(keypoints: np.ndarray) -> np.ndarray:
        """
        Get the polygon for the keypoints.
        """
        # Get the convex hull of the hands
        hull = ConvexHull(keypoints)
        return keypoints[hull.vertices]
    keypoints_mask = Image.new('L', image_size, 0)
    for keypoints in keypoints_list:
        # Generate polygon for left, right hands
        keypoints_polygon = _get_polygon(keypoints[:, :2])
        keypoints_coords = [(int(x), int(y)) for x, y in keypoints_polygon]
        # Draw polygon on mask
        draw = ImageDraw.Draw(keypoints_mask)
        draw.polygon(keypoints_coords, fill=255)
    return keypoints_mask

def expand_mask(mask: Image.Image, kernel_size: int = 3, iterations: int = 1) -> Image.Image:
    """
    Expand the mask using OpenCV's dilate function.
    
    :param mask: The input mask image.
    :param kernel_size: The size of the structuring element.
    :param iterations: The number of dilation iterations.
    :return: The expanded mask image.
    """
    # Convert the mask to a numpy array
    mask_array = np.array(mask)
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))  # 使用圆形结构元素
    # Perform dilation
    expanded_mask_array = cv2.dilate(mask_array, kernel, iterations=iterations)
    # Convert back to an Image
    expanded_mask = Image.fromarray(expanded_mask_array, mode='L')
    return expanded_mask

def get_crop_bbox(
    mask: Image.Image,
    padding: int = 100,
    min_size: int = 512,
    max_size: int = 768,
    tile_size: int = 32
) -> tuple:
    """
    Get the bounding box of the mask.

    :param mask: The input mask image.
    :param padding: The padding to add to the bounding box.
    :param min_size: The minimum size of the bounding box.
    :param max_size: The maximum size of the bounding box.
    :param tile_size: The tile size to use for the bounding box.
    :return: The bounding box(xyxy) of the mask.
    """
    mask_array = np.array(mask)
    width, height = mask_array.shape
    bbox = cv2.boundingRect(mask_array)
    x, y, w, h = bbox
    center_x, center_y = x + w // 2, y + h // 2
    w += padding * 2
    h += padding * 2
    w = min(max(w, min_size), min(max_size, width))
    h = min(max(h, min_size), min(max_size, height))
    w = (w + tile_size - 1) // tile_size * tile_size
    h = (h + tile_size - 1) // tile_size * tile_size
    x = max(0, center_x - w // 2)
    y = max(0, center_y - h // 2)
    if x + w > width:
        x = width - w
    if y + h > height:
        y = height - h
    return x, y, x + w, y + h

def deblur_image(image_path: str, output_path: str, prompt: str = None, target_height: int = 1536):
    image = Image.open(image_path)
    height = image.height
    width = image.width
    scale_h = target_height / height
    target_width = int(image.width * scale_h) // 16 * 16
    image = image.resize((target_width, target_height))

    if deblur_pipe is None:
        load_deblur_pipe()
    deblurred_image = deblur_pipe(
        prompt="" if prompt is None else prompt,
        image=image,
        num_inference_steps=30,
        strength=0.3,
        height=target_height,
        width=target_width,
        generator=torch.Generator().manual_seed(0)
    ).images[0]
    deblurred_image.save(output_path)
    return deblurred_image

def get_repair_parts_points(pose_parts: dict, repair_parts: str) -> dict:
    """
    Get the repair parts points from the pose parts.
    """
    repair_parts = repair_parts.split(',')
    decoded_parts = []
    for repair_part in repair_parts:
        assert repair_part in ['hands', 'left_hand', 'right_hand', 'feet', 'left_foot', 'right_foot']
        if repair_part == 'hands':
            decoded_parts.extend(['left_hand', 'right_hand'])
        elif repair_part == 'feet':
            decoded_parts.extend(['left_foot', 'right_foot'])
        else:
            decoded_parts.append(repair_part)

    repair_parts_points = {}
    for part in decoded_parts:
        if part == 'left_hand':
            points = np.vstack([pose_parts['hands'][0:21], pose_parts['bodies'][7]])
        elif part == 'right_hand':
            points = np.vstack([pose_parts['hands'][21:42], pose_parts['bodies'][4]])
        elif part == 'left_foot':
            points = pose_parts['feet'][0:3]
        elif part == 'right_foot':
            points = pose_parts['feet'][3:6]
        else:
            raise ValueError(f"Invalid repair part: {part}")
        repair_parts_points[part] = points
    return repair_parts_points

def repair_by_pose_parts(
    image_path: str,
    pose_path: str,
    output_path: str,
    fix_parts: str,
    mask_padding: int = 5,
    target_height: int = 1536,
    num_images_per_prompt: int = 1,
    prompt: str = "Masterpiece, High Definition, Real Person Portrait, 5 Fingers, Girl's Hand") -> None:
    # Load image and pose
    image = Image.open(image_path)
    name = os.path.basename(image_path).split('.')[0]
    height = image.height
    scale_h = target_height / height
    target_width = int(image.width * scale_h) // 16 * 16
    scale_w = target_width / image.width
    image = image.resize((target_width, target_height))
    pose = load_pose(pose_path)
    pose[:, 0] *= scale_w
    pose[:, 1] *= scale_h
    pose = to_openpose_format(pose)
    pose_parts = get_pose_parts(pose)
    # Draw hands on image
    repair_parts_points = get_repair_parts_points(pose_parts, fix_parts)
    for part, points in repair_parts_points.items():
        draw_image = draw_pose_on_image(image.copy(), points, color=(0, 255, 0))
        draw_image.save(os.path.join(output_path, f"{name}_{part}.png"))

    # Get hands mask
    keypoints_masks = get_keypoints_polygon_mask(repair_parts_points.values(), image.size)
    image_with_mask = image.copy()
    image_with_mask.paste(keypoints_masks, (0, 0), keypoints_masks)
    image_with_mask.save(os.path.join(output_path, f"{name}_image_with_mask.png"))

    expanded_keypoints_masks = expand_mask(keypoints_masks, kernel_size=8, iterations=mask_padding)
    expanded_keypoints_masks = expanded_keypoints_masks.filter(ImageFilter.GaussianBlur(radius=12))
    expanded_keypoints_masks.save(os.path.join(output_path, f"{name}_expanded_keypoints_masks_blur.png"))
    image_with_mask = image.copy()
    image_with_mask.paste(expanded_keypoints_masks, (0, 0), expanded_keypoints_masks)
    image_with_mask.save(os.path.join(output_path, f"{name}_image_with_expanded_mask.png"))
    
    # Inpaint 
    if inpaint_pipe is None:
        load_inpaint_pipe()
    seed = 42
    for i in range(num_images_per_prompt):
        for num_inference_steps in [15]:
            for guidance_scale in [15]:
                inpainted_images = inpaint_pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=expanded_keypoints_masks.convert('RGB'),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=image.height,
                    width=image.width,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=torch.Generator().manual_seed(seed)
                ).images
                for i, inpainted_image in enumerate(inpainted_images):
                    inpainted_image.save(os.path.join(output_path, f"{name}_inpainted_{i}_{seed}_{num_inference_steps}_{guidance_scale}.png"))
                    # Paste the inpainted image on the original image according to the expanded mask
                    paste_image = image.copy()
                    paste_image.paste(inpainted_image, (0, 0), expanded_keypoints_masks)
                    paste_image.save(os.path.join(output_path, f"{name}_inpainted_on_original_{i}_{seed}_{num_inference_steps}_{guidance_scale}.png"))

        import random
        seed = random.randint(0, 1000000)