"""
Scan unprocessed mimic motion tasks and infer them in the workspace with the following conditions:
1. character.png and character_pose.json exist
2. mimic_motion.mp4 not exist
"""

import argparse
import logging
import datetime
from pathlib import Path
from typing import List
import yaml
from concurrent.futures import ProcessPoolExecutor

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path("data/workspace")
TASK_DIR = WORKSPACE_DIR / "gens"
VIDEO_DIR = WORKSPACE_DIR / "videos"


def gen_mimic_motion_conf(
    tasks: List[Path],
    num_frames: int = 72,
    resolution: int = 576,
) -> str:
    """
    Generate the mimic motion configuration for the given tasks
    Args:
        tasks: List of tasks to generate the configuration for
        num_frames: Number of frames to generate
        resolution: Resolution of the generated video
    Returns:
        Path to the generated configuration file
    """
    conf_dir = Path(WORKSPACE_DIR) / "logs" / "mimic_motion"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    conf_path = conf_dir / conf_name

    config = {
        "base_model_path": "models/stable_video_diffusion",
        "ckpt_path": "models/MimicMotion/MimicMotion_1-1.pth",
        "test_case": []
    }

    for task in tasks:
        video_id = task.name
        test_case = {
            "video_pose_dir": str(VIDEO_DIR / video_id / "poses"),
            "ref_image_path": str(task / "character.png"),
            "ref_pose_path": str(task / "character_pose.json"),
            "output_path": str(task / "mimic_motion.mp4"),
            "num_frames": num_frames,
            "resolution": resolution,
            "frames_overlap": 6,
            "num_inference_steps": 25,
            "noise_aug_strength": 0,
            "guidance_scale": 3.0,
            "sample_stride": 1,
            "fps": 30,
            "seed": 42
        }
        config["test_case"].append(test_case)

    return config

def _run_task(gpu_id, start, length, task_conf, config_prefix):
    from modules.mimic_motion.inference import inference

    num_tasks = min(length, len(task_conf["test_case"]) - start)
    logger.info(f"Running {num_tasks} tasks on GPU {gpu_id}")
    config = task_conf.copy()
    config["test_case"] = config["test_case"][start:start+num_tasks]
    config_name = f"{config_prefix}_shard_{start}.yaml"
    config_path = Path(WORKSPACE_DIR) / "logs" / "mimic_motion" / config_name
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    with torch.cuda.device(gpu_id):
        with torch.no_grad():
            inference(config_path)

def run(tasks: list[Path], num_frames: int = 72, resolution: int = 576) -> None:
    """
    Run a list of task configurations in parallel.
    """
    logger.info(f"Found {len(tasks)} tasks")
    num_tasks = len(tasks)
    inference_conf = gen_mimic_motion_conf(tasks, num_frames, resolution)
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    num_gpus = torch.cuda.device_count()
    max_batch_size = 8
    logger.info(f"Number of available GPUs: {num_gpus}")

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        task_iter = 0
        futures = []
        while task_iter < num_tasks:
            batch_size = min(max_batch_size, (num_tasks - task_iter + num_gpus - 1) // num_gpus)
            for i in range(num_gpus):
                if task_iter < num_tasks:
                    futures.append(
                        executor.submit(_run_task, i, task_iter, batch_size, inference_conf, date_str))
                    task_iter += batch_size
        for future in futures:
            future.result()

def main(
    task_dir: str = "",
    character_id: str = "",
    video_id: str = "",
    skip_if_exists: bool = False,
    num_frames: int = 72,
    resolution: int = 576) -> None:
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
            if not (video_dir / "character.png").exists() or not (video_dir / "character_pose.json").exists():
                logger.info(f"Invalid task: {video_dir}")
                continue
            result_path = video_dir / "mimic_motion.mp4"
            if skip_if_exists and result_path.exists():
                logger.info(f"Task {video_dir} already exists, skipping")
                continue
            tasks.append(video_dir)
    run(tasks, num_frames, resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--task_dir", type=str, default="", help="Only run the task in the given task")
    parser.add_argument("--character_id", type=str, default="", help="Only run the task for the given character")
    parser.add_argument("--video_id", type=str, default="", help="Only run the task for the given video")
    parser.add_argument("--num_frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=576)
    args = parser.parse_args()
    main(args.task_dir, args.character_id, args.video_id, args.skip_if_exists, args.num_frames, args.resolution)