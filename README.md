# BetterAnimator

## Introduction

This is a simple tool to animate human motion.

## Features

- [x] Toon shading follow [Diffutoon](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- [ ] Animate human motion by [MimicMotion](https://github.com/Tencent/MimicMotion)
- [ ] Extract pose by [sapiens](https://github.com/facebookresearch/sapiens)
- [ ] Fix face by [FaceFusion](https://github.com/facefusion/facefusion)
- [ ] Consistent character generation with different poses
- [ ] Pose smoothing
- [ ] Manual pose editing

## Setup

### Environment setup

```bash
git submodule update --init --recursive
conda env create -f envs/toonshading_env.yaml
# or update the environment
conda env update -f envs/toonshading_env.yaml
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/MimicMotion
```

### Download models

```bash
bash scripts/download_models.sh
```



## References

- [MimicMotion](https://github.com/Tencent/MimicMotion)
- [FaceFusion](https://github.com/facefusion/facefusion)
- [Diffutoon](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- [sapiens](https://github.com/facebookresearch/sapiens)