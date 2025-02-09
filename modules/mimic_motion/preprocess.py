import os
from typing import List, Tuple

import numpy as np

from pose_estimation.utils import (
    load_pose, to_openpose_format, get_pose_parts)

def load_image_pose(pose_path: str) -> np.ndarray:
    pose = load_pose(pose_path)
    pose = to_openpose_format(pose)
    return pose

def load_video_pose(video_pose_dir: str) -> List[np.ndarray]:
    video_poses = []
    for file in sorted(os.listdir(video_pose_dir)):
        video_poses.append(load_image_pose(os.path.join(video_pose_dir, file)))
    return video_poses


def _get_alignment_keypoints_id(ref_pose_body: np.ndarray) -> List[int]:
    # Filter low confidence keypoints
    num_keypoints = ref_pose_body.shape[0]
    assert num_keypoints == 18, f"Number of keypoints in reference pose is not 18, but {num_keypoints}"
    candidate_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    valid_keypoints_id = [i for i in candidate_keypoint_id if ref_pose_body[i, 2] > 0.3]
    return valid_keypoints_id

def _pose_alignment(ref_body, video_bodys, width, height) -> Tuple[np.ndarray, np.ndarray]:
    num_video_poses = len(video_bodys)

    # Get valid alignment keypoints id
    alignment_keypoints_id = _get_alignment_keypoints_id(ref_body)
    ref_body = ref_body[alignment_keypoints_id]
    video_bodys = [body[alignment_keypoints_id] for body in video_bodys]
    video_bodys = np.stack(video_bodys)

    # Compute linear-rescale params
    # TODO(wanglong): normalize x,y before fitting
    x_ref, y_ref = ref_body[:, 0], ref_body[:, 1]
    x_video, y_video = video_bodys[:, 0], video_bodys[:, 1]
    ay, by = np.polyfit(y_video, np.tile(y_ref, num_video_poses), 1)
    ax = ay / (height / width)
    bx = np.mean(x_ref - x_video * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    return a, b

def _draw_pose(pose_parts, height, width, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.
    """
    scale = np.array([width, height])
    # Change to mimicmotion format
    bodies, bodies_score = pose_parts['bodies'][:, :2], pose_parts['bodies'][:, 2]
    bodies = bodies / scale
    subset = [i if bodies_score[i] > 0.3 else -1 for i in range(bodies.shape[0])]
    
    hands, hands_score = pose_parts['hands'][:, :2], pose_parts['hands_score']
    hands = hands / scale

    faces, faces_score = pose_parts['faces'][:, :2], pose_parts['faces_score']
    faces = faces / scale

    return draw_pose(dict())
    
    
    
    
    

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
        [pose['bodies'] for pose in video_poses_parts], ref_width, ref_height)
    for pose_parts in video_poses_parts:
        pose_parts['bodies'][:, :2] = pose_parts['bodies'][:, :2] * a + b
        pose_parts['hands'][:, :2] = pose_parts['hands'][:, :2] * a + b
        pose_parts['faces'][:, :2] = pose_parts['faces'][:, :2] * a + b
        pose_parts['foot'][:, :2] = pose_parts['foot'][:, :2] * a + b

    # Draw pose
    ref_pose_pixel = _draw_pose(ref_pose_parts, ref_height, ref_width)
    video_poses_pixels = [
        _draw_pose(pose_parts, ref_height, ref_width) for pose_parts in video_poses_parts]
    return np.stack([ref_pose_pixel] + video_poses_pixels)




