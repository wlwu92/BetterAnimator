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
        # generate_example_video_pose()

    def test_pose_alignment(self):
        image_pose = load_image_pose("data/mimicmotion_test/example_image_pose.json")
        video_poses = load_video_pose("data/mimicmotion_test/poses")
        ref_pose_body = get_pose_parts(image_pose)['bodies']
        video_pose_bodies = [get_pose_parts(pose)['bodies'] for pose in video_poses]
        a, b = _pose_alignment(ref_pose_body, video_pose_bodies)
        import pdb; pdb.set_trace()
        np.testing.assert_allclose(a, self.expected_a)
        np.testing.assert_allclose(b, self.expected_b)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

