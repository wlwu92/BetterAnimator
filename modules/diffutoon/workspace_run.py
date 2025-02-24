"""
Scan unprocessed diffutoon tasks and run them in the workspace with the following conditions:
1. diffutoon.mp4 not exists
2. face_fusion.mp4 exists
"""
import argparse
import json
from pathlib import Path
import logging
import torch
import ffmpeg
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path("data/workspace")
TASK_DIR = WORKSPACE_DIR / "gens"
CHARACTER_DIR = WORKSPACE_DIR / "characters"



def gen_config(task_dir: Path) -> dict:
    """
    Generate a config for a given task directory.
    """
    video_file = task_dir / "face_fusion.mp4"
    assert video_file.exists(), f"Video file not found: {video_file}"
    assert task_dir.parent.parent == TASK_DIR, f"Invalid task directory: {task_dir}"
    video_id = task_dir.name
    character_id = task_dir.parent.name
    character_dir = CHARACTER_DIR / character_id

    # Get prompt and negative prompt
    prompt_file = character_dir / "prompt.json"
    prompt = ""
    negative_prompt = ""
    with open(prompt_file, "r") as f:
        prompt_json = json.load(f)
        prompt = prompt_json["prompt"]
        negative_prompt = prompt_json["negative_prompt"]
        seed = prompt_json["seed"]

    # Get fps
    video_dir = WORKSPACE_DIR / "videos" / video_id
    video_info_file = video_dir / "video_info.json"
    with open(video_info_file, "r") as f:
        video_info = json.load(f)
        assert "r_frame_rate" in video_info, f"Invalid video info: {video_info}"
        r_frame_rate = video_info["r_frame_rate"]
        numerator, denominator = map(int, r_frame_rate.split("/"))
        fps = numerator / denominator
    height = 1536
    width = 896

    config = {
        "models": {
            "model_list": [
                "models/stable_diffusion/aingdiffusion_v12.safetensors",
                "models/AnimateDiff/mm_sd_v15_v2.ckpt",
                "models/ControlNet/control_v11f1e_sd15_tile.pth",
                "models/ControlNet/control_v11p_sd15_lineart.pth"
            ],
            "textual_inversion_folder": "models/textual_inversion",
            "device": "cuda",
            "lora_alphas": [],
            "controlnet_units": [
                {
                    "processor_id": "tile",
                    "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
                    "scale": 0.5
                },
                {
                    "processor_id": "lineart",
                    "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                    "scale": 0.5
                }
            ]
        },
        "data": {
            "input_frames": {
                "video_file": str(video_file),
                "image_folder": None,
                "height": height,
                "width": width,
                "start_frame_id": None,
                "end_frame_id": None
            },
            "controlnet_frames": [
                {
                    "video_file": str(video_file),
                    "image_folder": None,
                    "height": height,
                    "width": width,
                    "start_frame_id": None,
                    "end_frame_id": None
                },
                {
                    "video_file": str(video_file),
                    "image_folder": None,
                    "height": height,
                    "width": width,
                    "start_frame_id": None,
                    "end_frame_id": None
                }
            ],
            "output_folder": str(task_dir / "diffutoon"),
            "fps": fps
        },
        "pipeline": {
            "seed": seed,
            "pipeline_inputs": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_scale": 7.0,
                "clip_skip": 2,
                "denoising_strength": 1.0,
                "num_inference_steps": 10,
                "animatediff_batch_size": 16,
                "animatediff_stride": 8,
                "unet_batch_size": 1,
                "controlnet_batch_size": 1,
                "cross_frame_attention": False,
                # The following parameters will be overwritten. You don't need to modify them.
                "input_frames": [],
                "num_frames": 0,
                "height": height,
                "width": width,
                "controlnet_frames": []
            }
        }
    }
    return config


def _run_task(gpu_id, task_dir, task_conf):
    from diffsynth import SDVideoPipelineRunner
    runner = SDVideoPipelineRunner()
    logger.info(f"Running: {task_dir} on GPU {gpu_id}")
    with open(task_dir / "diffutoon_config.json", "w") as f:
        json.dump(task_conf, f, indent=4)
    with torch.cuda.device(gpu_id):
        runner.run(task_conf)
    # ffmpeg compress the output video
    result_file = task_dir / "diffutoon" / "video.mp4"
    output_file = task_dir / "diffutoon.mp4"
    stream = ffmpeg.input(str(result_file)).output(str(output_file))
    stream.run(overwrite_output=True, quiet=True)

def run(task_confs: list[tuple[Path, dict]]) -> None:
    """
    Run a list of task configurations in parallel.
    """

    logger.info(f"Running {len(task_confs)} tasks")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(_run_task, i % num_gpus, task_dir, task_conf)
            for i, (task_dir, task_conf) in enumerate(task_confs)]
        for future in futures:
            future.result()

def main(
    task_dir: str = "",
    skip_if_exists: bool = False,
    character_id: str = "",
    video_id: str = "",
):
    task_confs = []
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
            diffutoon_file = video_dir / "diffutoon.mp4"
            if diffutoon_file.exists() and skip_if_exists:
                logger.info(f"Diffutoon file already exists: {diffutoon_file}")
                continue
            face_fusion_file = video_dir / "face_fusion.mp4"
            if not face_fusion_file.exists():
                logger.info(f"Face fusion file not found: {face_fusion_file}")
                continue
            task_confs.append((video_dir, gen_config(video_dir)))
    run(task_confs)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="", help="Only run the task in the given task")
    parser.add_argument("--character_id", type=str, default="", help="Only run the task for the given character")
    parser.add_argument("--video_id", type=str, default="", help="Only run the task for the given video")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if the diffutoon file exists")
    args = parser.parse_args()
    main(args.task_dir, args.skip_if_exists, args.character_id, args.video_id)