"""
Scan unprocessed face fusion tasks and run them in the workspace with the following conditions:
1. mimic_motion.mp4 exists
2. face_fusion.mp4 not exists
"""

import argparse
import logging
from pathlib import Path
import subprocess

WORKSPACE_DIR = Path("data/workspace")
TASK_DIR = WORKSPACE_DIR / "gens"
CHARACTER_DIR = WORKSPACE_DIR / "characters"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run(task: Path) -> None:
    logger.info(f"Running: {task}")
    # Use original character image as source for better face reference
    character_id = task.parent.name
    character_dir = CHARACTER_DIR / character_id
    character_path = character_dir / "character.png"
    assert character_path.exists(), f"Character image not found: {character_path}"
    subprocess.run([
        "./facefusion.py",
        "headless-run",
        "-s", str(character_path.resolve()),
        "-t", str((task / "mimic_motion.mp4").resolve()),
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

def main(task_dir: str = "", skip_if_exists: bool = False) -> None:
    tasks = []
    if task_dir:
        task_dir = Path(task_dir)
        if not (task_dir / "mimic_motion.mp4").exists():
            logger.info(f"Missing mimic motion: {task_dir}")
            return
        if (task_dir / "face_fusion.mp4").exists() and skip_if_exists:
            logger.info(f"Skipping: {task_dir}")
            return
        tasks.append(task_dir)
    else:
        for character_dir in sorted(Path(TASK_DIR).glob("*")):
            for video_dir in sorted(character_dir.glob("*")):
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
    parser.add_argument("--task_dir", type=str, required=False, default="")
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()
    main(args.task_dir, args.skip_if_exists)