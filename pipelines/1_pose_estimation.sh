#!/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_image_dir|input_image_list> <output_pose_dir>"
    exit 1
fi

INPUT=$1
OUTPUT=$2

POSE_CHECKPOINT='models/sapiens/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2'
DETECTION_CONFIG_FILE='third_party/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT='models/sapiens/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

#---------------------------VISUALIZATION PARAMS--------------------------------------------------
LINE_THICKNESS=3 ## line thickness of the skeleton
RADIUS=6 ## keypoint radius
KPT_THRES=0.3 ## confidence threshold

##-------------------------------------inference-------------------------------------
# batch size 1 run faster
BATCH_SIZE=1

# Find all images and sort them, then write to a temporary text file
if [ -d "${INPUT}" ]; then
    IMAGE_LIST="${INPUT}/image_list.txt"
    find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"
else
    IMAGE_LIST="${INPUT}"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/sapiens/lite/demo:$(pwd)
RUN_FILE='third_party/sapiens/lite/demo/vis_pose.py'
# Bugfix following link: https://github.com/open-mmlab/mmdetection/issues/12008#issuecomment-2449327570
python -c "import experiments.hack_registry; exec(open('${RUN_FILE}').read())" \
    ${POSE_CHECKPOINT} \
    --num_keypoints 133 \
    --det-config ${DETECTION_CONFIG_FILE} \
    --det-checkpoint ${DETECTION_CHECKPOINT} \
    --batch-size ${BATCH_SIZE} \
    --input "${IMAGE_LIST}" \
    --output-root="${OUTPUT}" \
    --radius ${RADIUS} \
    --kpt-thr ${KPT_THRES}

rm "${IMAGE_LIST}"

echo "Processing complete."
echo "Results saved to $OUTPUT"
