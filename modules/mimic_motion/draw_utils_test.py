import unittest
import numpy as np
from PIL import Image

from modules.mimic_motion.draw_utils import draw_pose
from modules.mimic_motion.preprocess import load_image_pose
from pose_estimation.utils import get_pose_parts

class TestDrawPose(unittest.TestCase):
    def setUp(self):
        self.image_pose = load_image_pose('data/mimicmotion_test/example_image_pose.json')
        self.image = Image.open('data/mimicmotion_test/example_image.png')

    def test_draw_pose(self):
        width, height = self.image.size
        pose_parts = get_pose_parts(self.image_pose)
        image = draw_pose(pose_parts, height, width)
        image = Image.fromarray(image)
        image.save('data/mimicmotion_test/example_image_pose.png')

if __name__ == '__main__':
    unittest.main()
