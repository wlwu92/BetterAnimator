"""
Scan unprocessed face fusion tasks and run them in the workspace with the following conditions:
1. mimic_motion.mp4 exists
2. face_fusion.mp4 not exists
"""

import argparse
import ffmpeg
import logging
import os
import time
from pathlib import Path
import subprocess

import torch

WORKSPACE_DIR = Path("data/workspace")
TASK_DIR = WORKSPACE_DIR / "gens"
CHARACTER_DIR = WORKSPACE_DIR / "characters"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def upscale_video(video_file: str, output_file: str) -> None:
    # Current video is: 576x1024
    # Upscale to: 864x1536
    probe = ffmpeg.probe(video_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_file}")
    video = ffmpeg.input(video_file)
    video = video.filter("scale", "864", "-1", flags="lanczos")
    video.output(output_file, vcodec='libx264', crf=18, preset='slow').run(overwrite_output=True, quiet=True)

def run(task: Path, gpu_id: int = 0) -> subprocess.Popen:
    logger.info(f"Running: {task} on GPU {gpu_id}")
    input_dir = task / "2_mimic_motion"
    output_dir = task / "3_face_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    upscale_video(str(input_dir / "mimic_motion.mp4"), str(output_dir / "mimic_motion_upscaled.mp4"))
    # Use original character image as source for better face reference
    character_id = task.parent.name
    character_dir = CHARACTER_DIR / character_id
    character_path = character_dir / "character.png"
    assert character_path.exists(), f"Character image not found: {character_path}"
    return subprocess.Popen([
        "./facefusion.py",
        "headless-run",
        "-s", str(character_path.resolve()),
        "-t", str((output_dir / "mimic_motion_upscaled.mp4").resolve()),
        "-o", str((output_dir / "face_fusion.mp4").resolve()),
        "--processors", "face_swapper", "expression_restorer", "face_enhancer",
        "--face-detector-model", "scrfd",
        "--face-detector-score", "0.2",
        "--face-detector-angles", "0", "90", "180",
        "--face-selector-mode", "one",
        "--face-mask-types", "occlusion",
        "--face-occluder-model", "xseg_2",
        "--execution-providers", "cuda",
        "--temp-path", str(output_dir.resolve()),
        "--keep-temp"
    ],
        cwd="third_party/facefusion",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    )

def run_tasks(tasks: list[Path]) -> None:
    num_tasks = len(tasks)
    num_gpus = torch.cuda.device_count()

    processors = {}
    current_iter = 0
    while current_iter < num_tasks:
        for i in range(num_gpus):
            if i not in processors or processors[i].poll() is not None:
                if current_iter < num_tasks:
                    processors[i] = run(tasks[current_iter], i)
                    current_iter += 1
        time.sleep(0.1)
    for processor in processors.values():
        try:
            processor.wait()
        except Exception as e:
            logger.error(f"Error waiting for processor: {e}")

def main(
    task_dir: str = "",
    character_id: str = "",
    video_id: str = "",
    skip_if_exists: bool = False) -> None:
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
            input_dir = video_dir / "2_mimic_motion"
            output_dir = video_dir / "3_face_fusion"
            if not (input_dir / "mimic_motion.mp4").exists():
                logger.info(f"Missing mimic motion: {input_dir}")
                continue
            if output_dir.exists() and skip_if_exists:
                logger.info(f"Skipping: {output_dir}")
                continue
            tasks.append(video_dir)
    logger.info(f"Found {len(tasks)} tasks")
    run_tasks(tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False, default="", help="Only run the task in the given task")
    parser.add_argument("--character_id", type=str, required=False, default="", help="Only run the task for the given character")
    parser.add_argument("--video_id", type=str, required=False, default="", help="Only run the task for the given video")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if the face fusion file exists")
    args = parser.parse_args()
    main(args.task_dir, args.character_id, args.video_id, args.skip_if_exists)