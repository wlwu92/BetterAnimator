import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pose_estimation.utils import (
    load_pose, to_openpose_format, get_pose_parts)
from modules.mimic_motion.draw_utils import draw_pose

def load_image_pose(pose_path: str) -> np.ndarray:
    pose = load_pose(pose_path)
    pose = to_openpose_format(pose)
    return pose

def load_video_pose(video_pose_dir: str) -> List[np.ndarray]:
    video_poses = []
    video_pose_dir = Path(video_pose_dir)
    for file in sorted(video_pose_dir.glob("*.json"), key=lambda x: int(x.stem)):
        video_poses.append(load_image_pose(file))
    return video_poses


def _get_alignment_keypoints_id(ref_pose_body: np.ndarray) -> List[int]:
    # Filter low confidence keypoints
    num_keypoints = ref_pose_body.shape[0]
    assert num_keypoints == 18, f"Number of keypoints in reference pose is not 18, but {num_keypoints}"
    candidate_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    valid_keypoints_id = [i for i in candidate_keypoint_id if ref_pose_body[i, 2] > 0.3]
    return valid_keypoints_id

def _pose_alignment(ref_body, video_bodies) -> Tuple[np.ndarray, np.ndarray]:
    num_video_poses = len(video_bodies)

    # Get valid alignment keypoints id
    alignment_keypoints_id = _get_alignment_keypoints_id(ref_body)
    ref_body = ref_body[alignment_keypoints_id]
    video_bodies = [body[alignment_keypoints_id] for body in video_bodies]
    video_bodies = np.concatenate(video_bodies, axis=0)

    # Compute linear-rescale params
    x_ref, y_ref = ref_body[:, 0], ref_body[:, 1]
    x_video, y_video = video_bodies[:, 0], video_bodies[:, 1]
    ay, by = np.polyfit(y_video, np.tile(y_ref, num_video_poses), 1)
    ax = ay
    bx = np.mean(np.tile(x_ref, num_video_poses) - x_video * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    return a, b

def generate_pose_pixels(ref_pose, video_poses, ref_width, ref_height):
    """
    Generate pose pixels from video poses and image pose.
    """
    # Get parts
    ref_pose_parts = get_pose_parts(ref_pose)
    video_poses_parts = [get_pose_parts(pose) for pose in video_poses]

    # Pose alignment
    a, b = _pose_alignment(
        ref_pose_parts['bodies'],
        [pose['bodies'] for pose in video_poses_parts])
    for pose_parts in video_poses_parts:
        pose_parts['bodies'][:, :2] = pose_parts['bodies'][:, :2] * a + b
        pose_parts['hands'][:, :2] = pose_parts['hands'][:, :2] * a + b
        pose_parts['faces'][:, :2] = pose_parts['faces'][:, :2] * a + b
        pose_parts['foot'][:, :2] = pose_parts['foot'][:, :2] * a + b

    # Draw pose
    # [H, W, 3]
    ref_pose_pixel = draw_pose(ref_pose_parts, ref_height, ref_width)
    video_poses_pixels = [
        draw_pose(pose_parts, ref_height, ref_width) for pose_parts in video_poses_parts]
    # [N, 3, H, W]
    return np.stack([ref_pose_pixel] + video_poses_pixels).transpose(0, 3, 1, 2)
