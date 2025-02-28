import click
import logging
import os
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.character_generation.repair_hands import repair_by_pose_parts, deblur_image

WORKSPACE_DIR = "data/workspace/"
CHARACTER_DIR = os.path.join(WORKSPACE_DIR, "characters")

@click.group()
def main():
    pass


@main.command()
@click.option('--image_path', type=str, required=True, help='Image path or image directory or image list separated by comma')
@click.option('--output_dir', type=str, required=True, help='Output directory')
@click.option('--prompt', type=str, default=None, help='Prompt')
def deblur(image_path, output_dir, prompt):
    """
    Repair images using the specified image path, pose path, and output directory.
    """
    if os.path.isdir(image_path):
        image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
        image_list.sort()
    elif os.path.isfile(image_path):
        image_list = [image_path]
    else:
        image_list = image_path.split(',')
        for image_path in image_list:
            if not os.path.exists(image_path):
                logger.error(f"Image path {image_path} does not exist")
                return
    logger.info(f"Processing {len(image_list)} images")
    os.makedirs(output_dir, exist_ok=True)
    for image_path in image_list:
        load_prompt = prompt
        if prompt is None:
            image_name = os.path.basename(image_path)
            if "_ref_" in image_name and "_frame_" in image_name:
                ref_id = int(image_name.split("_ref_")[1].split("_frame_")[0])
                prompt_path = os.path.join(CHARACTER_DIR, f"{ref_id:06d}", "prompt.json")
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r') as f:
                        prompt_info = json.load(f)
                        if "prompt_deblur" in prompt_info:
                            prompt = prompt_info["prompt_deblur"]
                            logger.info(f"Use prompt_deblur: {prompt}")
                        else:
                            logger.info(f"No prompt_deblur in {prompt_path}")
        logger.info(f"Processing image: {image_path}")
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        deblur_image(image_path, output_path, load_prompt)
        
        
@main.command()
@click.option('--image_path', type=str, required=True, help='Image path or image directory')
@click.option('--pose_path', type=str, required=True, help='Pose path or pose directory')
@click.option('--output_dir', type=str, required=True, help='Path to the output file')
@click.option('--fix_parts', type=str, default="hands", help='Parts to fix, e.g. hands, left_hand, right_hand, feet, left_foot, right_foot')
@click.option('--mask_padding', type=int, default=5, help='Padding of the mask')
@click.option('--target_height', type=int, default=1536, help='Target height of the output image')
@click.option('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt')
@click.option('--prompt', type=str, default="Masterpiece, High Definition, Real Person Portrait, 5 Fingers, Girl's Hand", help='Prompt')
def repair(image_path, pose_path, output_dir, fix_parts, mask_padding, target_height, num_images_per_prompt, prompt):
    """
    Repair images using the specified image path, pose path, and output directory.
    """
    if os.path.isdir(image_path):
        image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
        image_list.sort()
    else:
        image_list = [image_path]
    if os.path.isdir(pose_path):
        pose_list = [os.path.join(pose_path, f) for f in os.listdir(pose_path) if f.endswith('.json')]
        pose_list.sort()
    else:
        pose_list = [pose_path]
    if len(image_list) != len(pose_list):
        logger.error("The number of image paths and pose paths must match.")
        return
    
    logger.info(f"Processing {len(image_list)} images, save to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for image_path, pose_path in zip(image_list, pose_list):
        logger.info(f"Processing image: {image_path} and pose: {pose_path}")
        repair_by_pose_parts(image_path, pose_path, output_dir, fix_parts, mask_padding=mask_padding, target_height=target_height, num_images_per_prompt=num_images_per_prompt, prompt=prompt)

if __name__ == "__main__":
    main()