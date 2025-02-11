import click
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.character_generation.repair_hands import repair_hands
from modules.character_generation.generate_character_scales import generate_character_scales

WORKSPACE_DIR = "data/workspace/"
CHARACTER_DIR = os.path.join(WORKSPACE_DIR, "characters")

@click.group()
def main():
    pass

@main.command()
@click.option('--image_path', type=str, required=True, help='Character image path')
@click.option('--character_id', type=str, default=None, help='Character id')
def add_character(image_path, character_id):
    """
    Add a character to the character database and generate different scale by outpainting.
    """
    # Generate character id from image path
    if character_id is None:
        character_id = os.path.basename(image_path).split('.')[0]
    # Create character directory
    character_dir = os.path.join(CHARACTER_DIR, character_id)
    os.makedirs(character_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(character_dir, "character.png"))
    generate_character_scales(character_dir)

@main.command()
@click.option('--image_path', type=str, required=True, help='Image path or image directory')
@click.option('--pose_path', type=str, required=True, help='Pose path or pose directory')
@click.option('--output_dir', type=str, required=True, help='Path to the output file')
@click.option('--target_height', type=int, default=1536, help='Target height of the output image')
def repair(image_path, pose_path, output_dir, target_height):
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
        repair_hands(image_path, pose_path, output_dir, target_height)

if __name__ == "__main__":
    main()