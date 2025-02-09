import json
import numpy as np
from PIL import Image, ImageDraw

def load_pose(pose_path: str) -> np.ndarray:
    """
    Load a pose from a JSON file.

    Args:
        pose_path (str): The path to the JSON file containing the pose.

    Returns:
        np.ndarray: A numpy array containing the pose.
    """
    with open(pose_path, 'r') as file:
        data = json.load(file)

    assert len(data['instance_info']) == 1, "Only one instance is supported"
    keypoints = data['instance_info'][0]['keypoints']
    keypoint_scores = data['instance_info'][0]['keypoint_scores']
    pose_array = np.array([kp + [score] for kp, score in zip(keypoints, keypoint_scores)])
    return pose_array

def to_openpose_format(pose_array: np.ndarray) -> np.ndarray:
    """
    Convert the pose array to the OpenPose format.
    """
    assert pose_array.shape[0] == 133, "Pose array must have 133 elements"
    neck = np.mean(pose_array[[5, 6]], axis=0)
    # neck score
    neck[2] = np.logical_and(pose_array[5, 2] > 0.3, pose_array[6, 2] > 0.3).astype(int)
    pose_array = np.insert(pose_array, 17, neck, axis=0)
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    pose_array[openpose_idx] = pose_array[mmpose_idx]
    return pose_array

def get_pose_parts(pose_array: np.ndarray) -> dict:
    """
    Get the parts of the pose from the pose array.
    """
    bodies = pose_array[:18]
    hands = pose_array[92:113]
    hands = np.vstack([hands, pose_array[113:]])
    faces = pose_array[24:92]
    foot = pose_array[18:24]
    return dict(bodies=bodies, hands=hands, faces=faces, foot=foot)


def draw_pose_on_image(image: Image.Image, pose_array: np.ndarray, color: tuple | None = None) -> Image.Image:
    """
    Draw the pose on the image with color indicating score.
    """
    draw = ImageDraw.Draw(image)
    for point in pose_array:
        x, y, score = point
        if color is None:
            red = int(255 * score)
            green = int(255 * (1 - score))
            color = (red, green, 0)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=color, outline=color)
    return image