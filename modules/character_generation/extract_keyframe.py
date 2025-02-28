"""
Extract keyframe following the steps:
    1. Get key_frame_id from video_info.json
    2. Get key_frame_image from task_dir/3_face_fusion/facefusion/mimic_motion_upscaled/{key_frame_id + 1:8d}.png
    3. Copy key_frame_image to task_dir/4_keyframe/keyframe_origin.png
    4. Copy task_dir/4_keyframe/keyframe_origin.png to manual_dir/5_keyframe_labeling/{character_id}_{video_id}/keyframe.png
    5. Deblur keyframe.png as keyframe_deblur_{id}.png
    6. Manual select the best deblurred image as keyframe.png
    7. If keyframe is not good, select another keyframe from the same video and repeat the steps above
"""

import json
import shutil
from pathlib import Path
import argparse
import logging
import subprocess
from typing import List
import tempfile
import random

from common.constant import TASK_DIR, VIDEO_DIR, MANUAL_DIR
from modules.character_generation.repair_hands import deblur_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def deblur_keyframes(keyframe_paths: List[Path]) -> None:
    for keyframe_path in keyframe_paths:
        task_dir = keyframe_path.parent.parent
        video_id, character_id = task_dir.name, task_dir.parent.name
        output_dir = MANUAL_DIR / "keyframe_deblur_selection" / f"{character_id}_{video_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            seed = random.randint(0, 1000000)
            deblur_image(keyframe_path, output_dir / f"deblur_{i}_{seed}.png", seed=seed)


def extract_keyframe(task_dir: str, character_id: str, video_id: str, skip_if_exists: bool = True) -> None:
    tasks = []
    if task_dir:
        assert character_id == "" and video_id == "", "task_dir is not allowed to be used with character_id or video_id"
        task_dir = Path(task_dir)
        video_id = task_dir.name
        character_id = task_dir.parent.name
    candidate_characters = sorted(TASK_DIR.glob("*")) \
        if not character_id else [TASK_DIR / character_id]
    collected_keyframes = []
    for character_dir in candidate_characters:
        candidate_videos = sorted(character_dir.glob("*")) \
            if not video_id else [character_dir / video_id]
        for video_dir in candidate_videos:
            output_dir = video_dir / "4_keyframe"
            if output_dir.exists() and skip_if_exists:
                logging.info(f"Key frame already processed: {output_dir}")
                continue
            output_dir.mkdir(parents=True)
            video_id = video_dir.name
            video_info_path = VIDEO_DIR / video_id / "video_info.json"
            if not video_info_path.exists():
                logger.info(f"Video info not found: {video_dir}")
                continue
            video_info = json.load(open(video_info_path))
            if "key_frame_id" not in video_info:
                logger.info(f"Key frame id not found in video info: {video_info_path}")
                continue
            key_frame_id = video_info["key_frame_id"]
            
            input_dir = video_dir / "3_face_fusion" / "facefusion" / "mimic_motion_upscaled"
            key_frame_image = input_dir / f"{key_frame_id + 1:08d}.png"
            if not key_frame_image.exists():
                logger.info(f"Key frame image not found: {video_dir}")
                continue

            shutil.copy(key_frame_image, output_dir / "keyframe_origin.png")
            collected_keyframes.append(output_dir / "keyframe_origin.png")
    if collected_keyframes:
        logger.info(f"Deblurring {len(collected_keyframes)} keyframes")
        deblur_keyframes(collected_keyframes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--character_id", type=str, required=False)
    parser.add_argument("--video_id", type=str, required=False)
    parser.add_argument("--skip_if_exists", type=bool, default=True)
    args = parser.parse_args()
    extract_keyframe(args.task_dir, args.character_id, args.video_id, args.skip_if_exists)
