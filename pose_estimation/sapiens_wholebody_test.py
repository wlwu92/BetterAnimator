import unittest
from PIL import Image

from pose_estimation.sapiens_wholebody import detect
from pose_estimation.utils import load_pose, to_openpose_format, draw_pose_on_image

class SapiensWholebodyTest(unittest.TestCase):
    def test_sapiens_wholebody(self):
        image_path = "data/original.png"
        detect(image_path, "data/original_sapiens_wholebody.json")
        pose = load_pose("data/original_sapiens_wholebody.json")
        pose = to_openpose_format(pose)
        image = draw_pose_on_image(Image.open(image_path), pose)
        image.save("data/original_sapiens_wholebody.png")


if __name__ == "__main__":
    unittest.main()
