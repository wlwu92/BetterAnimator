import os
import argparse
import shutil
import logging

from common.constant import CHARACTER_DIR
from modules.character_generation.generate_character_scales import generate_character_scales

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def add_character(
    image_path: str | None = None,
    character_id: str | None = None,
    num_inference_steps: int = 2,
    scale: str | None = None) -> None:
    """
    Add a character to the character database and generate different scale by outpainting.
    """
    # Generate character id from image path
    if character_id is None:
        characters = sorted(CHARACTER_DIR.iterdir())
        character_id = 0 if len(characters) == 0 else int(characters[-1].name) + 1
        character_id = f"{character_id:06d}"
        character_dir = CHARACTER_DIR / character_id
        if character_dir.exists():
            logger.error(f"Character directory {character_dir} already exists")
            return
        character_dir.mkdir(parents=True)
    else:
        character_dir = CHARACTER_DIR / character_id
        if not character_dir.exists():
            logger.error(f"Character directory {character_dir} does not exist")
            return
    character_image_path = character_dir / "character.png"
    if image_path is not None:
        shutil.copy(image_path, character_image_path)
    assert character_image_path.exists(), f"Character image {character_image_path} does not exist"
    generate_character_scales(character_dir, num_inference_steps=num_inference_steps, update_scale=scale)

def main():
    parser = argparse.ArgumentParser(description='Add a character to the character database and generate different scales by outpainting.')
    parser.add_argument('--image_path', type=str, default=None, help='Character image path')
    parser.add_argument('--character_id', type=str, default=None, help='Character ID')
    parser.add_argument('--num_inference_steps', type=int, default=2, help='Number of inference steps')
    parser.add_argument('--scale', type=str, default=None, help='Scale to update, e.g. x1, x2, x3')
    args = parser.parse_args()
    add_character(args.image_path, args.character_id, args.num_inference_steps, args.scale)

if __name__ == "__main__":
    main()
