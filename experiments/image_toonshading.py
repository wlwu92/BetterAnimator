import argparse
import subprocess
import os
import shutil
import random
import tempfile

import cv2

def convert_image_to_video(image_file, output_video_file, fps=30):
    command = [
        'ffmpeg',
        '-y',
        '-loop', '1',
        '-i', image_file,
        '-c:v', 'libx264',
        '-t', '1',
        '-pix_fmt', 'yuv420p',
        '-vf', f'fps={fps}',
        output_video_file
    ]
    subprocess.run(command, check=True)

def process_video(video_file, prompt, output_folder, seed):
    from diffsynth import SDVideoPipelineRunner, download_models

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_file}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width // 64 * 64
    height = height // 64 * 64
    cap.release()

    download_models([
        "AingDiffusion_v12",
        "AnimateDiff_v2",
        "ControlNet_v11p_sd15_lineart",
        "ControlNet_v11f1e_sd15_tile",
        "TextualInversion_VeryBadImageNegative_v1.3"
    ])
    image_folder = None
    num_frames = 1

    config = {
        "models": {
            "model_list": [
                "models/stable_diffusion/aingdiffusion_v12.safetensors",
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
                "video_file": video_file,
                "image_folder": image_folder,
                "height": height,
                "width": width,
                "start_frame_id": 0,
                "end_frame_id": num_frames
            },
            "controlnet_frames": [
                {
                    "video_file": video_file,
                    "image_folder": image_folder,
                    "height": height,
                    "width": width,
                    "start_frame_id": 0,
                    "end_frame_id": num_frames
                },
                {
                    "video_file": video_file,
                    "image_folder": image_folder,
                    "height": height,
                    "width": width,
                    "start_frame_id": 0,
                    "end_frame_id": num_frames
                }
            ],
            "output_folder": output_folder,
            "fps": 30
        },
        "pipeline": {
            "seed": seed,
            "pipeline_inputs": {
                "prompt": prompt,
                "negative_prompt": "verybadimagenegative_v1.3, embroidery, printed patterns, graphic design elements",
                "cfg_scale": 7.0,
                "clip_skip": 2,
                "denoising_strength": 1.0,
                "num_inference_steps": 10,
                "animatediff_batch_size": 16,
                "animatediff_stride": 8,
                "unet_batch_size": 1,
                "controlnet_batch_size": 1,
                "cross_frame_attention": False,
                "input_frames": [],
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "controlnet_frames": []
            }
        }
    }

    runner = SDVideoPipelineRunner()
    runner.run(config)

def save_generated_image(image_file, gen_image, prompt, seed):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]
    gen_dir = os.path.join(os.path.dirname(image_file), f"{image_name}_gen")
    os.makedirs(gen_dir, exist_ok=True)
    
    existing_files = os.listdir(gen_dir)
    
    max_id = 0
    for file in existing_files:
        if file.endswith(".png"):
            try:
                file_id = int(os.path.splitext(file)[0].split('_')[-1])
                if file_id > max_id:
                    max_id = file_id
            except ValueError:
                continue
    gen_id = max_id + 1
    gen_image_file = os.path.join(gen_dir, f"gen_{gen_id}.png")
    shutil.copy(gen_image, gen_image_file)
    prompt_file = os.path.join(gen_dir, f"prompt_{gen_id}_seed_{seed}.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

def main():
    parser = argparse.ArgumentParser(description="Process a video with a given prompt.")
    parser.add_argument('image_file', type=str, help='Path to the image file')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for the random number generator')
    
    args = parser.parse_args()
    
    prompt_file_path = os.path.join(os.path.dirname(args.image_file), 'prompt.txt')
    
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read().strip()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_file = os.path.join(temp_dir, "temp.mp4")
        tmp_output_folder = os.path.join(temp_dir, "output")
        convert_image_to_video(args.image_file, temp_video_file)
        if args.seed == -1:
            seed = random.randint(0, 1000000)
        else:
            seed = args.seed
        # TODO: Use image pipeline instead of video pipeline
        process_video(temp_video_file, prompt, tmp_output_folder, seed)
        os.remove(temp_video_file)
        gen_image = os.path.join(tmp_output_folder, "frames", "0.png")
        save_generated_image(args.image_file, gen_image, prompt, seed)


if __name__ == "__main__":
    main()
