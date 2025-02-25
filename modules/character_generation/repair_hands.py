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
    inpaint_pipe = flux_fill_pipe(lora_name="models/FLUX/F.1_FitnessTrainer_lora_v1.0.safetensors", enable_multi_gpu=MULTI_DEVICE_INFERENCE)

from pose_estimation.utils import (
    load_pose,
    draw_pose_on_image,
    get_pose_parts,
    to_openpose_format,
)

def get_hands_mask(hands: np.ndarray, image_size: tuple) -> Image.Image:
    """
    Get the hands mask from the pose parts.
    """
    def _get_polygon(hands: np.ndarray) -> np.ndarray:
        """
        Get the polygon for the hands.
        """
        # Get the convex hull of the hands
        hull = ConvexHull(hands)
        return hands[hull.vertices]
    hands_mask = Image.new('L', image_size, 0)
    # Generate polygon for left, right hands
    left_hands = hands[0:21, :2].astype(np.int32)
    right_hands = hands[21:42, :2].astype(np.int32)
    left_hand_polygon = _get_polygon(left_hands)
    right_hand_polygon = _get_polygon(right_hands)
    left_hand_coords = [(int(x), int(y)) for x, y in left_hand_polygon]
    right_hand_coords = [(int(x), int(y)) for x, y in right_hand_polygon]
    # Draw polygon on mask
    draw = ImageDraw.Draw(hands_mask)
    draw.polygon(left_hand_coords, fill=255)
    draw.polygon(right_hand_coords, fill=255)
    return hands_mask

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

def repair_hands(image_path: str, pose_path: str, output_path: str, target_height: int = 1536) -> None:
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
    draw_image = draw_pose_on_image(image.copy(), pose_parts['hands'], color=(0, 255, 0))
    draw_image.save(os.path.join(output_path, f"{name}_hands.png"))

    # Get hands mask
    hands_mask = get_hands_mask(pose_parts['hands'], image.size)
    hands_mask.save(os.path.join(output_path, f"{name}_hands_mask.png"))
    image_with_mask = image.copy()
    image_with_mask.paste(hands_mask, (0, 0), hands_mask)
    image_with_mask.save(os.path.join(output_path, f"{name}_image_with_mask.png"))

    expanded_hands_mask = expand_mask(hands_mask, kernel_size=8, iterations=5)
    expanded_hands_mask.save(os.path.join(output_path, f"{name}_expanded_hands_mask.png"))
    expanded_hands_mask = expanded_hands_mask.filter(ImageFilter.GaussianBlur(radius=5))
    expanded_hands_mask.save(os.path.join(output_path, f"{name}_expanded_hands_mask_blur.png"))
    image_with_mask = image.copy()
    image_with_mask.paste(expanded_hands_mask, (0, 0), expanded_hands_mask)
    image_with_mask.save(os.path.join(output_path, f"{name}_image_with_expanded_mask.png"))
    
    # Inpaint hands
    if inpaint_pipe is None:
        load_inpaint_pipe()
    inpainted_image = inpaint_pipe(
        # prompt="Masterpiece, High Definition, Real Person Portrait, 5 Fingers, Girl's Hand",
        prompt="",
        image=image,
        mask_image=expanded_hands_mask.convert('RGB'),
        num_inference_steps=20,
        guidance_scale=30,
        height=image.height,
        width=image.width,
        generator=torch.Generator().manual_seed(0)
    ).images[0]
    inpainted_image.save(os.path.join(output_path, f"{name}.png"))