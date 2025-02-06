#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <video_path>"
    exit 1
fi

video_path=$1
VIDEOS_DIR="${WORKSPACE_ROOT}/videos"
MAX_ID=$(ls ${VIDEOS_DIR} | tail -n 1)
NEW_ID=$(printf "%06d" $((MAX_ID + 1)))
echo "Adding video to workspace with id ${NEW_ID}"

# Add video to workspace
if [ -d "${VIDEOS_DIR}/${NEW_ID}" ]; then
    echo "Video with id ${NEW_ID} already exists"
    exit 1
fi
mkdir -p ${VIDEOS_DIR}/${NEW_ID}
DST_VIDEO_PATH="${VIDEOS_DIR}/${NEW_ID}/video.mp4"
cp ${video_path} ${DST_VIDEO_PATH}

# Extract frames
mkdir -p ${VIDEOS_DIR}/${NEW_ID}/frames
ffmpeg -i ${DST_VIDEO_PATH} ${VIDEOS_DIR}/${NEW_ID}/frames/%06d.png