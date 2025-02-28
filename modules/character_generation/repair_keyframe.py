"""
Repair keyframe following the steps:
    1. 4_keyframe/keyframe_deblur.png exists
    2. generate 4_keyframe/keyframe_deblur_pose.json
    3. generate repaired keyframes in manual_dir/keyframe_repair_selection/{character_id}_{video_id}/
    4. manual select the best repaired keyframe as 4_keyframe/keyframe_repaired.png
    5. If all repaired keyframes are not good, modify the prompt and repeat the steps above
"""

import json
import shutil
from pathlib import Path
import argparse
import logging
from typing import List
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from common.constant import TASK_DIR, VIDEO_DIR, MANUAL_DIR
from modules.character_generation.repair_hands import repair_by_pose_parts
from pose_estimation.sapiens_wholebody import detect as sapiens_detect


def repair_keyframe(input_file: Path, output_dir: Path) -> None:
    task_dir = input_file.parent.parent
    pose_file = input_file.parent / input_file.name.replace(".png", "_pose.json")
    sapiens_detect(input_file, pose_file)
    video_id = task_dir.name
    character_id = task_dir.parent.name
    output_dir = MANUAL_DIR / "keyframe_repair_selection" / f"{character_id}_{video_id}"
    output_dir.mkdir(parents=True)

    video_info_path = VIDEO_DIR / video_id / "video_info.json"
    with open(video_info_path, "r") as f:
        video_info = json.load(f)
    repair_parts = video_info.get("repair_parts", "hands")
    mask_padding = video_info.get("mask_padding", 10)
    foot_mask_padding = video_info.get("foot_mask_padding", mask_padding)
    hand_prompt = video_info.get("hand_repair_prompt", "Masterpiece, High Definition, Real Person Portrait, 5 Fingers, Girl's Hand")
    foot_prompt = video_info.get("foot_repair_prompt", "detailed high-resolution shoes, clear shoes texture, realistic footwear, sharp edges, photorealistic shoes with fine details")
    repair_parts = [part.strip() for part in repair_parts.split(",")]
    hands_parts = [part for part in repair_parts if part in ["hands", "left_hand", "right_hand"]]
    feet_parts = [part for part in repair_parts if part in ["feet", "left_foot", "right_foot"]]
    for i in range(20):
        seed = random.randint(0, 1000000)
        parts_str = ",".join(hands_parts)
        logger.info(f"Repairing hands parts: {parts_str}, mask padding: {mask_padding}, prompt: {hand_prompt}, seed: {seed}")
        repaired_image = repair_by_pose_parts(
            input_file,
            pose_file,
            output_dir,
            parts_str,
            mask_padding=mask_padding,
            prompt=hand_prompt,
            seed=seed
        )
        parts_str = ",".join(feet_parts)
        logger.info(f"Repairing feet parts: {parts_str}, mask padding: {foot_mask_padding}, prompt: {foot_prompt}, seed: {seed}")
        repaired_image = repair_by_pose_parts(
            repaired_image,
            pose_file,
            output_dir,
            parts_str,
            mask_padding=foot_mask_padding,
            prompt=foot_prompt,
            seed=seed
        )

def main(task_dir: str, character_id: str, video_id: str, skip_if_exists: bool = True) -> None:
    tasks = []
    if task_dir:
        assert character_id == "" and video_id == "", "task_dir is not allowed to be used with character_id or video_id"
        task_dir = Path(task_dir)
        video_id = task_dir.name
        character_id = task_dir.parent.name
    candidate_characters = sorted(TASK_DIR.glob("*")) \
        if not character_id else [TASK_DIR / character_id]
    for character_dir in candidate_characters:
        candidate_videos = sorted(character_dir.glob("*")) \
            if not video_id else [character_dir / video_id]
        for video_dir in candidate_videos:
            input_file = video_dir / "4_keyframe" / "keyframe_deblur.png"
            if not input_file.exists():
                logger.info(f"Key frame deblur image not found: {video_dir}")
                continue
            labeled_file = video_dir / "4_keyframe" / "keyframe_repair.png"
            if labeled_file.exists() and skip_if_exists:
                logging.info(f"Key frame already repaired: {labeled_file}")
                continue
            video_id = video_dir.name
            character_id = video_dir.parent.name
            output_dir = MANUAL_DIR / "keyframe_repair_selection" / f"{character_id}_{video_id}"
            if output_dir.exists() and skip_if_exists:
                logging.info(f"Repairing task already exists: {output_dir}")
                continue
            logger.info(f"Repairing keyframe: {video_dir}")
            repair_keyframe(input_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--character_id", type=str, required=False)
    parser.add_argument("--video_id", type=str, required=False)
    parser.add_argument("--skip_if_exists", type=bool, default=True)
    args = parser.parse_args()
    main(args.task_dir, args.character_id, args.video_id, args.skip_if_exists)
