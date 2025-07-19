# BetterAnimator

## Introduction

This is a simple tool to animate human motion.

## Features

- [x] Toon shading follow [Diffutoon](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- [x] Animate human motion by [MimicMotion](https://github.com/Tencent/MimicMotion)
- [x] Extract pose by [sapiens](https://github.com/facebookresearch/sapiens)
- [ ] Fix face by [FaceFusion](https://github.com/facefusion/facefusion)
- [ ] Consistent character generation with different poses
- [ ] Pose smoothing
- [ ] Manual pose editing

## Setup

### Environment setup

#### Linux

```bash
git submodule update --init --recursive
conda env create -f envs/toonshading_env.yaml
# or update the environment
conda env update -f envs/toonshading_env.yaml
conda activate toonshading
# Install sapiens dependencies
bash envs/toonshadingPostBuild.sh
# Setup PYTHONPATH before running the code
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/MimicMotion
```

#### MacOS

```bash
git submodule update --init --recursive
conda env create -f envs/toonshading_osx_env.yaml
conda activate toonshading
# Some operators are not supported on MPS, so we need to enable MPS fallback
# https://github.com/pytorch/pytorch/issues/77764
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

* Flux on MacOS
    * [diffusers](https://github.com/huggingface/diffusers) runs slowly(12 mins for 1 image with gguf Q4_K_S).
    * [mflux](https://github.com/filipstrand/mflux) has alot of limitations
        * only support text-to-image, image-to-image, controlnet(canny only) generation
    * Waiting for mflux to support controlnet, inpaint and more features.

### Download models

```bash
bash scripts/download_models.sh
```

## Run

```shell
# 1. add video and extract pose and gen tasks
REFERENCE_DIR=data/workspace/tmp/20250719/references
CHARACTERS=(000001 000006 000020)
for i in `ls ${REFERENCE_DIR}/*`; do
    ID=$(basename $i .mp4)
    bash pipelines/0_add_video.sh $i $ID
    bash pipelines/1_pose_estimation.sh data/workspace/videos/${ID}/frames data/workspace/videos/${ID}/poses
    for C_ID in $CHARACTERS; do
        python modules/mimic_motion/match_character_scale.py --video_id $ID --character_id $C_ID
    done
done  
# 2. Run mimic motion
python modules/mimic_motion/workspace_run.py
# 3. Run face fusion
python modules/face_fusion/workspace_run.py
# 4. Run diffutoon
python modules/diffutoon/workspace_run.py
```


## References

- [MimicMotion](https://github.com/Tencent/MimicMotion)
- [FaceFusion](https://github.com/facefusion/facefusion)
- [Diffutoon](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- [sapiens](https://github.com/facebookresearch/sapiens)