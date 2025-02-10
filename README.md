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
```
* [diffusers](https://github.com/huggingface/diffusers) runs slowly(12 mins for 1 image with gguf Q4_K_S).
* [mflux](https://github.com/filipstrand/mflux) has alot of limitations
    * only support text-to-image, image-to-image, controlnet(canny only) generation
* Waiting for mflux to support controlnet, inpaint and more features.

### Download models

```bash
bash scripts/download_models.sh
```



## References

- [MimicMotion](https://github.com/Tencent/MimicMotion)
- [FaceFusion](https://github.com/facefusion/facefusion)
- [Diffutoon](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- [sapiens](https://github.com/facebookresearch/sapiens)