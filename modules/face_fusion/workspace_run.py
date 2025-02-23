"""
Scan unprocessed face fusion tasks and run them in the workspace with the following conditions:
1. mimic_motion.mp4 exists
2. face_fusion.mp4 not exists
"""

import argparse
import logging
from pathlib import Path
import subprocess
import ffmpeg

WORKSPACE_DIR = Path("data/workspace")
TASK_DIR = WORKSPACE_DIR / "gens"
CHARACTER_DIR = WORKSPACE_DIR / "characters"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def upscale_video(video_file: str, output_file: str) -> None:
    # Current video is: 576x1024
    # Upscale to: 896 x {height} and then crop to 896x1536
    probe = ffmpeg.probe(video_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_file}")
    input_width, input_height = int(video_stream['width']), int(video_stream['height'])
    dst_height = int(input_height * 896 / input_width)
    start_y = max(0, (dst_height - 1536) // 2)
    height = min(1536, dst_height)
    if height != 1536:
        logger.error(f"Upscaling {video_file} to {height}x896")
    video = ffmpeg.input(video_file)
    video = video.filter("scale", "896", "-1", flags="lanczos").crop(0, start_y, 896, height)
    video.output(output_file, vcodec='libx264', crf=18, preset='slow').run(overwrite_output=True, quiet=True)

def run(task: Path) -> None:
    logger.info(f"Running: {task}")
    upscale_video(str(task / "mimic_motion.mp4"), str(task / "mimic_motion_upscaled.mp4"))
    # Use original character image as source for better face reference
    character_id = task.parent.name
    character_dir = CHARACTER_DIR / character_id
    character_path = character_dir / "character.png"
    assert character_path.exists(), f"Character image not found: {character_path}"
    subprocess.run([
        "./facefusion.py",
        "headless-run",
        "-s", str(character_path.resolve()),
        "-t", str((task / "mimic_motion_upscaled.mp4").resolve()),
        "-o", str((task / "face_fusion.mp4").resolve()),
        "--processors", "face_swapper", "expression_restorer", "face_enhancer",
        "--face-detector-model", "scrfd",
        "--face-detector-score", "0.2",
        "--face-detector-angles", "0", "90", "180",
        "--face-selector-mode", "one",
        "--face-mask-types", "occlusion",
        "--face-occluder-model", "xseg_2",
        "--execution-providers", "cuda",
    ], cwd="third_party/facefusion")

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
            if not (video_dir / "mimic_motion.mp4").exists():
                logger.info(f"Missing mimic motion: {video_dir}")
                continue
            if (video_dir / "face_fusion.mp4").exists() and skip_if_exists:
                logger.info(f"Skipping: {video_dir}")
                continue
            tasks.append(video_dir)
    logger.info(f"Found {len(tasks)} tasks")
    for task in tasks:
        run(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=False, default="", help="Only run the task in the given task")
    parser.add_argument("--character_id", type=str, required=False, default="", help="Only run the task for the given character")
    parser.add_argument("--video_id", type=str, required=False, default="", help="Only run the task for the given video")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if the face fusion file exists")
    args = parser.parse_args()
    main(args.task_dir, args.character_id, args.video_id, args.skip_if_exists)