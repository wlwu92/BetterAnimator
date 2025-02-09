# Copy from https://github.com/Tencent/MimicMotion/blob/main/inference.py

import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit

from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.transforms.functional import to_pil_image

# Third party dependencies
from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4

from modules.mimic_motion.preprocess import (
    load_image_pose, load_video_pose, generate_pose_pixels
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_pose_dir, ref_image_path, ref_pose_path, resolution=576):
    image_pixels = pil_loader(ref_image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    new_w, new_h = resolution, int(resolution * h / w)
    assert new_w % 64 == 0 and new_h % 64 == 0, f"new_w: {new_w}, new_h: {new_h}"
    image_pixels = resize(image_pixels, [new_h, new_w], antialias=None)
    image_pixels = image_pixels.permute(1, 2, 0).numpy()
    image_pixels = np.transpose(image_pixels, (0, 3, 1, 2))
    image_pixels = torch.from_numpy(image_pixels) / 127.5 - 1

    image_pose = load_image_pose(ref_pose_path)
    video_poses = load_video_pose(video_pose_dir)
    pose_pixels = generate_pose_pixels(
        video_poses, image_pose, width=new_w, height=new_h,
        ref_scale=new_h / h
    )
    pose_pixels = torch.from_numpy(pose_pixels) / 127.5 - 1
    return image_pixels, pose_pixels

def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(infer_config, device)
    import pdb; pdb.set_trace()
    for task in infer_config.test_case:
        ############################################## Pre-process data ##############################################
        image_pixels, pose_pixels = preprocess(task.video_pose_dir, task.ref_image_path, task.ref_pose_path, task.resolution)
        ########################################### Run MimicMotion pipeline ###########################################
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        ################################### save results to output folder. ###########################################
        Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
        save_to_mp4(
            _video_frames, 
            task.output_path,
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/mimicmotion/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")