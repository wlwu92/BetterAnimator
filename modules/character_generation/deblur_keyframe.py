"""
Extract and deblur keyframe following the steps:
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
from typing import List, Tuple
import random
from concurrent.futures import ProcessPoolExecutor
import torch
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from common.constant import TASK_DIR, VIDEO_DIR, MANUAL_DIR, CHARACTER_DIR
from modules.character_generation.repair_hands import deblur_image


def deblur_keyframe(keyframe_path: Path) -> None:
    task_dir = keyframe_path.parent.parent
    video_id, character_id = task_dir.name, task_dir.parent.name
    output_dir = MANUAL_DIR / "keyframe_deblur_selection" / f"{character_id}_{video_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = CHARACTER_DIR / character_id / "prompt.json"
    with open(prompt_file, "r") as f:
        prompt = json.load(f).get("deblur_prompt", "")
    for i in range(20):
        seed = random.randint(0, 1000000)
        deblur_image(keyframe_path, output_dir / f"deblur_{i}_{seed}.png", prompt=prompt, seed=seed)

def _run_tasks_on_gpu(gpu_id, gpu_tasks):
    with torch.cuda.device(gpu_id):
        logger.info(f"GPU {gpu_id} processing {len(gpu_tasks)} tasks")
        for video_dir, keyframe_path in gpu_tasks:
            dst_keyframe_path = video_dir / "4_keyframe" / "keyframe.png"
            dst_keyframe_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(keyframe_path, dst_keyframe_path)
            deblur_keyframe(dst_keyframe_path)

def batch_run(tasks: List[Tuple[Path, Path]]) -> None:
    num_tasks = len(tasks)
    logger.info(f"Found {num_tasks} tasks")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        tasks_per_gpu = [tasks[i::num_gpus] for i in range(num_gpus)] 
        futures = [executor.submit(_run_tasks_on_gpu, gpu_id, gpu_tasks) 
                    for gpu_id, gpu_tasks in enumerate(tasks_per_gpu) if gpu_tasks]
        for future in futures:
            future.result()

def extract_keyframe(task_dir: str, character_id: str, video_id: str, skip_if_exists: bool = True) -> None:
    if task_dir:
        assert character_id == "" and video_id == "", "task_dir is not allowed to be used with character_id or video_id"
        task_dir = Path(task_dir)
        video_id = task_dir.name
        character_id = task_dir.parent.name
    candidate_characters = sorted(TASK_DIR.glob("*")) \
        if not character_id else [TASK_DIR / character_id]
    deblur_tasks = []
    for character_dir in candidate_characters:
        candidate_videos = sorted(character_dir.glob("*")) \
            if not video_id else [character_dir / video_id]
        for video_dir in candidate_videos:
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

            output_dir = video_dir / "4_keyframe"
            if output_dir.exists() and skip_if_exists:
                logging.info(f"Key frame already processed: {output_dir}")
                continue
            deblur_tasks.append((video_dir, key_frame_image))
    batch_run(deblur_tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--character_id", type=str, required=False)
    parser.add_argument("--video_id", type=str, required=False)
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if the deblurring task already exists")
    args = parser.parse_args()
    extract_keyframe(args.task_dir, args.character_id, args.video_id, args.skip_if_exists)
