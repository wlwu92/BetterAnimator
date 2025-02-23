import shutil
import argparse
import logging
from pathlib import Path

import numpy as np

import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize

# Third party dependencies
from mimicmotion.utils.utils import save_to_mp4

from modules.mimic_motion.preprocess import (
    load_image_pose, load_video_pose, generate_pose_pixels
)

CHARACTER_DIR = "data/workspace/characters"
VIDEO_DIR = "data/workspace/videos"
TASK_DIR = "data/workspace/gens"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def match_character_scale(video_pose_dir, ref_image_path, ref_pose_path, resolution=576):
    image_pixels = pil_loader(ref_image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    new_w, new_h = resolution, int(resolution * h / w)
    assert new_w % 64 == 0 and new_h % 64 == 0, f"new_w: {new_w}, new_h: {new_h}"
    image_pixels = resize(image_pixels, [new_h, new_w], antialias=None)
    image_pixels = image_pixels.permute(1, 2, 0).numpy()
    image_pixels = np.transpose(image_pixels, (2, 0, 1))

    image_pose = load_image_pose(ref_pose_path)
    video_poses = load_video_pose(video_pose_dir)
    image_pose[:, :2] = image_pose[:, :2] * new_h / h
    pose_pixels = generate_pose_pixels(image_pose, video_poses, new_w, new_h, draw_feet=True)

    body_bbox_overlaps = np.zeros((new_h, new_w))
    for pose_pixel in pose_pixels:
        pose_pixel = pose_pixel.transpose(1, 2, 0)
        row_sum = np.sum(pose_pixel, axis=(1, 2))
        col_sum = np.sum(pose_pixel, axis=(0, 2))
        min_y, max_y = np.where(row_sum > 0)[0][0], np.where(row_sum > 0)[0][-1]
        min_x, max_x = np.where(col_sum > 0)[0][0], np.where(col_sum > 0)[0][-1]
        body_bbox_overlaps[min_y:max_y, min_x:max_x] += 1

    xys = np.argwhere(body_bbox_overlaps > (len(pose_pixels) * 0.1))
    bbox = np.array([xys[:, 1].min(), xys[:, 0].min(), xys[:, 1].max(), xys[:, 0].max()])

    for pose_pixel in pose_pixels:
        mask = np.sum(pose_pixel, axis=0) == 0
        pose_pixel[:, mask] = image_pixels[:, mask]

    if bbox[0] >= 0.0 * new_w and bbox[1] >= 0.05 * new_h and bbox[2] <= 1.0 * new_w and bbox[3] <= 0.95 * new_h:
        return pose_pixels, True
    return pose_pixels, False

def add_gen_task(video_id: str, character_id: str, skip_if_exists: bool = False) -> None:
    character_dirs = []
    video_dirs = []
    if character_id == "all":
        character_dirs = sorted(Path(CHARACTER_DIR).glob("*"))
    else:
        character_dirs = [Path(CHARACTER_DIR) / character_id]
    if video_id == "all":
        video_dirs = sorted(Path(VIDEO_DIR).glob("*"))
    else:
        video_dirs = [Path(VIDEO_DIR) / args.video_id]

    for character_dir in character_dirs:
        for video_dir in video_dirs:
            character_id = character_dir.stem
            video_id = video_dir.stem
            video_pose_dir = video_dir / "poses"
            task_dir = Path(TASK_DIR) / character_id / video_id
            if skip_if_exists and task_dir.exists():
                if (task_dir / "character.png").exists() \
                    and (task_dir / "character_pose.json").exists():
                    logger.info(f"Task {task_dir} already exists, skipping")
                    continue
            task_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Matching character {character_id} and video {video_id}")
            scale_files = sorted(character_dir.glob("character_x*.png"))
            for scale_file in scale_files:
                scale_pose = scale_file.parent / "poses" / (scale_file.stem + ".json")
                pose_pixels, is_matched = match_character_scale(str(video_pose_dir), str(scale_file), str(scale_pose))
                preview_video_path = task_dir / f"{scale_file.stem}.mp4"
                save_to_mp4(torch.tensor(pose_pixels), preview_video_path, fps=30)
                if is_matched:
                    shutil.copy(scale_file, task_dir / "character.png")
                    shutil.copy(scale_pose, task_dir / "character_pose.json")
                    break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--character_id", type=str, required=True)
    parser.add_argument("--skip_if_exists", action="store_true")

    args = parser.parse_args()
    add_gen_task(args.video_id, args.character_id, args.skip_if_exists)