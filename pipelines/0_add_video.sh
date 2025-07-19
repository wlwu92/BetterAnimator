#!/bin/bash
WORKSPACE_ROOT="data/workspace"

# Check arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <video_path> [id]"
    echo "  video_path: Path to the video file"
    echo "  id: Optional custom id (6-digit format, e.g., 000001). If not provided, auto-generated"
    exit 1
fi

video_path=$1
custom_id=$2
VIDEOS_DIR="${WORKSPACE_ROOT}/videos"

# Determine the id to use
if [ -n "${custom_id}" ]; then
    # Use custom id if provided
    NEW_ID=${custom_id}
    echo "Using custom id: ${NEW_ID}"
else
    # Auto-generate id
    MAX_ID=$(ls ${VIDEOS_DIR} 2>/dev/null | tail -n 1)
    if [ -z "${MAX_ID}" ]; then
        NEW_ID="000000"
    else
        NEW_ID=$(printf "%06d" $((MAX_ID + 1)))
    fi
    echo "Auto-generated id: ${NEW_ID}"
fi

# Check if video with this id already exists
if [ -d "${VIDEOS_DIR}/${NEW_ID}" ]; then
    echo "Video with id ${NEW_ID} already exists, skipping..."
    exit 0
fi

echo "Adding video to workspace with id ${NEW_ID}"

# Add video to workspace
mkdir -p ${VIDEOS_DIR}/${NEW_ID}
DST_VIDEO_PATH="${VIDEOS_DIR}/${NEW_ID}/video.mp4"
cp ${video_path} ${DST_VIDEO_PATH}

# Extract frames
mkdir -p ${VIDEOS_DIR}/${NEW_ID}/frames
ffmpeg -loglevel quiet -i ${DST_VIDEO_PATH} ${VIDEOS_DIR}/${NEW_ID}/frames/%06d.png
ffprobe -v quiet -print_format json -show_format -show_streams \
${video_path} \
| jq '{filename: .format.filename, width: .streams[0].width, height: .streams[0].height, r_frame_rate: .streams[0].r_frame_rate, duration: .streams[0].duration, nb_frames: .streams[0].nb_frames}' > ${VIDEOS_DIR}/${NEW_ID}/video_info.json