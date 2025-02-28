import unittest
from PIL import Image

from pose_estimation.dwpose_wholebody import DwposeWholebody
from pose_estimation.utils import to_openpose_format, draw_pose_on_image

class DwposeWholebodyTest(unittest.TestCase):
    def test_dwpose_wholebody(self):
        dwpose_detector = DwposeWholebody(
            model_det="models/DWPose/yolox_l.onnx",
            model_pose="models/DWPose/dw-ll_ucoco_384.onnx",
            device="cpu")
        image_path = "data/original.png"
        pose = dwpose_detector.detect(image_path)
        self.assertEqual(pose.shape, (1, 133, 3))
        image = Image.open(image_path)
        pose1 = dwpose_detector.detect(image)
        self.assertEqual(pose1.shape, (1, 133, 3))

        image = draw_pose_on_image(image, pose[0])
        image.save("data/original_dwpose_wholebody.png")

if __name__ == "__main__":
    unittest.main()