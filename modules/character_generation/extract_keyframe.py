"""
Extract keyframe following the steps:
    1. Get key_frame_id from video_info.json
    2. Get key_frame_image from task_dir/face_fusion/mimic_motion_upscaled/{key_frame_id:8d}.png
    3. Copy key_frame_image to task_dir/keyframe.png
    4. Backup task_dir/* (ignore keyframe.png, backup folder) to task_dir/backup/
    5. Rename task_dir/keyframe.png to task_dir/character.png
    6. Generate chatacter_pose.json
"""

import json
import shutil
from pathlib import Path
import argparse
import logging
import subprocess
from typing import List
import tempfile

from modules.character_generation.repair_hands import repair_hands

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORKSPACE = Path("data/workspace")
TASK_DIR = WORKSPACE / "gens"
VIDEO_DIR = WORKSPACE / "videos"

def process_task(video_dir: Path, key_frame_image: Path) -> None:
    shutil.copy(key_frame_image, video_dir / "keyframe.png")
    backup_dir = video_dir / "backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in video_dir.iterdir():
        if item.name in ["keyframe.png", "backup"]:
            continue
        shutil.move(item, backup_dir / item.name)

def batch_pose_estimation(keyframe_paths: List[Path]) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        for keyframe_path in keyframe_paths:
            video_id, character_id = keyframe_path.parent.name, keyframe_path.parent.parent.name
            shutil.copy(keyframe_path, temp_dir / f"{video_id}_{character_id}.png")
        pose_dir = temp_dir / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "bash", "pipelines/1_pose_estimation.sh",
            str(temp_dir),
            str(pose_dir),
        ])
        for keyframe_path in keyframe_paths:
            video_id, character_id = keyframe_path.parent.name, keyframe_path.parent.parent.name
            shutil.move(pose_dir / f"{video_id}_{character_id}.json", keyframe_path.parent / "keyframe_pose.json")
            shutil.move(pose_dir / f"{video_id}_{character_id}.png", keyframe_path.parent / "keyframe_pose.png")

def run_repair_hands(keyframe_paths: List[Path]) -> None:
    for keyframe_path in keyframe_paths:
        pose_path = keyframe_path.parent / "keyframe_pose.json"
        image_path = keyframe_path.parent / "keyframe.png"
        character_path = keyframe_path.parent / "character.png"
        repaired_dir = keyframe_path.parent / "repaired"
        repaired_dir.mkdir(parents=True, exist_ok=True)
        repair_hands(image_path, pose_path, repaired_dir, 1536)
        shutil.move(repaired_dir / "keyframe.png", character_path)

def extract_keyframe(task_dir: str, character_id: str, video_id: str) -> None:
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
            if (video_dir / "keyframe.png").exists():
                logger.info(f"Keyframe already exists: {video_dir}")
                collected_keyframes.append(video_dir / "keyframe.png")
                continue
            diffutoon_result = video_dir / "diffutoon.mp4"
            if not diffutoon_result.exists():
                logger.info(f"Diffutoon result not found: {video_dir}")
                continue
            face_fusion_dir = video_dir / "facefusion" / "mimic_motion_upscaled"
            if not face_fusion_dir.exists():
                logger.info(f"Face fusion not found: {video_dir}")
                continue
            video_info_path = VIDEO_DIR / video_dir.name / "video_info.json"
            if not video_info_path.exists():
                logger.info(f"Video info not found: {video_dir}")
                continue
            video_info = json.load(open(video_info_path))
            if "key_frame_id" not in video_info:
                logger.info(f"Key frame id not found in video info: {video_info_path}")
                continue
            key_frame_id = video_info["key_frame_id"]
            key_frame_image = face_fusion_dir / f"{key_frame_id + 1:08d}.png"
            if not key_frame_image.exists():
                logger.info(f"Key frame image not found: {video_dir}")
                continue
            logger.info(f"Processing {video_dir}")
            process_task(video_dir, key_frame_image)
            collected_keyframes.append(video_dir / "keyframe.png")
    if collected_keyframes:
        batch_pose_estimation(collected_keyframes)
        run_repair_hands(collected_keyframes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--character_id", type=str, required=False)
    parser.add_argument("--video_id", type=str, required=False)
    args = parser.parse_args()
    extract_keyframe(args.task_dir, args.character_id, args.video_id)
