import unittest

import numpy as np
from PIL import Image

from modules.mimic_motion.preprocess import load_image_pose, load_video_pose, _pose_alignment
from pose_estimation.utils import get_pose_parts

EXAMPLE_VIDEO_PATH = "data/mimicmotion_test/example_video.mp4"
REF_IMAGE_PATH = "data/mimicmotion_test/example_image.png"

def generate_example_video_pose():
    from pose_estimation.dwpose_wholebody import DwposeWholebody
    from pose_estimation.utils import save_pose
    dwpose = DwposeWholebody(
        model_det="models/DWPose/yolox_l.onnx",
        model_pose="models/DWPose/dw-ll_ucoco_384.onnx",
    )
    ref_pose = dwpose.detect(REF_IMAGE_PATH)
    save_pose(ref_pose[0], "data/mimicmotion_test/example_image_pose.json")
    import cv2
    cap = cv2.VideoCapture(EXAMPLE_VIDEO_PATH)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        pose = dwpose.detect(Image.fromarray(frame))
        save_pose(pose[0], f"data/mimicmotion_test/poses/{i}.json")

def get_mimicmotion_alignment_result():    
    return [1.14420872, 1.14420872], [-0.0278025, -0.10101006]
    # add `return a, b` in mimicmotion.dwpose.preprocess.get_video_pose
    # from mimicmotion.dwpose.preprocess import get_video_pose
    # image = Image.open(REF_IMAGE_PATH).convert("RGB")
    # image_pixels = np.array(image)
    # a, b = get_video_pose(EXAMPLE_VIDEO_PATH, image_pixels, sample_stride=1)
    # print(a, b)

class PreprocessTest(unittest.TestCase):
    
    def setUp(self):
        self.expected_a, self.expected_b = get_mimicmotion_alignment_result()
        self.ref_width, self.ref_height = Image.open(REF_IMAGE_PATH).size
        self.video_width, self.video_height = 1080, 1920
        # generate_example_video_pose()

    def test_pose_alignment(self):
        image_pose = load_image_pose("data/mimicmotion_test/example_image_pose.json")
        video_poses = load_video_pose("data/mimicmotion_test/poses")
        ref_pose_body = get_pose_parts(image_pose)['bodies']
        video_pose_bodies = [get_pose_parts(pose)['bodies'] for pose in video_poses]
        a, b = _pose_alignment(ref_pose_body, video_pose_bodies)
        # Normalized a/b to unormalized a/b
        # (x / width * ax + bx) * dst_width -> x1
        # (y / height * ay + by) * dst_height -> y1
        # x * (ax / width * dst_width) + bx * dst_width -> x1
        # y * (ay / height * dst_height) + by * dst_height -> y1
        # ax / width * dst_width = ay / height * dst_height
        dst_ay = self.expected_a[1] / self.video_height * self.ref_height
        dst_ax = dst_ay
        dst_by = self.expected_b[1] * self.ref_height
        dst_bx = self.expected_b[0] * self.ref_width
        print(a, b)
        print(dst_ax, dst_ay, dst_bx, dst_by)
        np.testing.assert_allclose(a[0], dst_ax, atol=1e-3)
        np.testing.assert_allclose(a[1], dst_ay, atol=1e-3)
        np.testing.assert_allclose(b[0], dst_bx, atol=1e0)
        np.testing.assert_allclose(b[1], dst_by, atol=1e0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

