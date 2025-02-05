#!/bin/bash

# Download mimic motion models
mkdir -p models/DWPose
if [ ! -f models/DWPose/yolox_l.onnx ]; then
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
fi
if [ ! -f models/DWPose/dw-ll_ucoco_384.onnx ]; then
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
fi

mkdir -p models/MimicMotion
if [ ! -f models/MimicMotion/MimicMotion_1-1.pth ]; then
    wget -P models/MimicMotion https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth
fi

# SVD model: stabilityai/stable-video-diffusion-img2vid-xt-1-1
echo "Make sure you have logged in huggingface and have access to download the model"
mkdir -p models/stable_video_diffusion
huggingface-cli download --resume-download --local-dir ./models/stable_video_diffusion stabilityai/stable-video-diffusion-img2vid-xt-1-1


# sapiens model
mkdir -p models/sapiens
sapiens_pose_model=sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2
sapiens_detector_model=rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
if [ ! -f models/sapiens/$sapiens_pose_model ]; then
    echo "Downloading sapiens pose model"
    huggingface-cli download \
        noahcao/sapiens-pose-coco \
        $sapiens_pose_model \
        --local-dir ./models/sapiens
fi

if [ ! -f models/sapiens/$sapiens_detector_model ]; then
    echo "Downloading sapiens detector model"
    huggingface-cli download \
        facebook/sapiens-pose-bbox-detector \
        $sapiens_detector_model \
        --local-dir ./models/sapiens
fi



