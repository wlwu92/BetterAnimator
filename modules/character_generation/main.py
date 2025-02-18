import click
import logging
import os
import shutil
import json

from modules.character_generation.animate_image import animate_image, animate_image_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.character_generation.repair_hands import repair_hands, deblur_image
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
        characters = sorted(os.listdir(CHARACTER_DIR))
        character_id = 0 if len(characters) == 0 else int(characters[-1]) + 1
        character_id = f"{character_id:06d}"
    character_dir = os.path.join(CHARACTER_DIR, character_id)
    if os.path.exists(character_dir):
        logger.error(f"Character directory {character_dir} already exists")
        return
    os.makedirs(character_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(character_dir, "character.png"))
    # Create character directory
    generate_character_scales(character_dir)

@main.command()
@click.option('--character_id', type=str, required=True, help='Character id')
@click.option('--image_path', type=str, default=None, help='Character image path')
@click.option('--update_scale', type=str, default=None, help='Update scale, e.g. x1, x2, x3')
@click.option('--num_inference_steps', type=int, default=2, help='Number of inference steps')
def update_character(character_id, image_path, update_scale, num_inference_steps):
    """
    Update a character in the character database and generate different scale by outpainting.
    """
    character_dir = os.path.join(CHARACTER_DIR, character_id)
    assert os.path.exists(character_dir), f"Character directory {character_dir} does not exist"
    if image_path is not None:
        shutil.copy(image_path, os.path.join(character_dir, "character.png"))
    # Create character directory
    generate_character_scales(
        character_dir, update_scale=update_scale, num_inference_steps=num_inference_steps)


@main.command()
@click.option('--image_path', type=str, required=True, help='Image path')
@click.option("--prompt", type=str, required=True, help="Prompt file path or prompt text")
@click.option('--output_path', type=str, default="", help='Output path')
@click.option('--seed', type=int, default=-1, help='Seed')
@click.option('--times', type=int, default=1, help='Number of times to random seed')
@click.option('--debug', '-d', is_flag=True, help='Debug mode')
def animate_exp(image_path, prompt, output_path, seed, times, debug):
    """
    Animate a character image.
    """
    negative_prompt = "embroidery, printed patterns, graphic design elements"
    if os.path.isfile(prompt):
        with open(prompt, 'r') as f:
            prompt_info = json.load(f)
            prompt = prompt_info["prompt"]
            if "seed" in prompt_info:
                print(f"Use seed in prompt file: {prompt_info['seed']}")
                seed = prompt_info["seed"]
            if "negative_prompt" in prompt_info:
                negative_prompt = prompt_info["negative_prompt"]
    if not output_path:
        if debug:
            output_dir = image_path.split('.')[0] + "_anime"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, image_path.split('/')[-1])
            import datetime
            time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_path.replace(".png", f"_{time_str}.png")
            prompt_save_path = output_path.replace(".png", "_prompt.json")
            with open(prompt_save_path, 'w') as f:
                json.dump(
                    {"prompt": prompt, "seed": seed, "negative_prompt": negative_prompt},
                    f,
                    indent=4)
        else:
            output_path = image_path.split('.')[0] + "_anime.png"
    if seed != -1:
        assert times == 1, "Times must be 1 when seed is not -1"
        animate_image(image_path, prompt, negative_prompt, output_path, seed=seed)
    else:
        animate_image_random_seed(image_path, prompt, negative_prompt, output_path, times=times)


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