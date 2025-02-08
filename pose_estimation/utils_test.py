import unittest
import numpy as np
from PIL import Image

from pose_estimation.utils import (
    load_pose, to_openpose_format, draw_pose_on_image, get_pose_parts
)

class TestLoadPose(unittest.TestCase):

    def setUp(self):
        self.test_pose_path = 'data/example_wholebody_pose.json'
        self.test_image_path = 'data/example_wholebody_image.png'

    def test_load_pose(self):
        pose_array = load_pose(self.test_pose_path)
        self.assertEqual(pose_array.shape[1], 3)
        self.assertEqual(pose_array.shape[0], 133)
        self.assertIsInstance(pose_array, np.ndarray)

    def test_to_openpose_format(self):
        pose_array = load_pose(self.test_pose_path)
        openpose_format = to_openpose_format(pose_array)
        self.assertEqual(openpose_format.shape[1], 3)
        self.assertEqual(openpose_format.shape[0], 134)

    def test_draw_pose(self):
        pose_array = load_pose(self.test_pose_path)
        image = Image.open(self.test_image_path)
        draw_image0 = draw_pose_on_image(pose_array, image.copy())
        draw_image0.save('data/example_wholebody_image_draw_pose.png')

        openpose_format = to_openpose_format(pose_array)
        draw_image1 = draw_pose_on_image(openpose_format, image.copy())
        draw_image1.save('data/example_wholebody_image_draw_openpose.png')
        self.assertIsInstance(image, Image.Image)

    def test_get_pose_parts(self):
        pose_array = load_pose(self.test_pose_path)
        pose_array = to_openpose_format(pose_array)
        pose_parts = get_pose_parts(pose_array)
        self.assertIsInstance(pose_parts, dict)
        self.assertEqual(len(pose_parts), 4)
        self.assertEqual(pose_parts['bodies'].shape[0], 18)
        self.assertEqual(pose_parts['hands'].shape[0], 42)
        self.assertEqual(pose_parts['faces'].shape[0], 68)
        self.assertEqual(pose_parts['foot'].shape[0], 6)

        image = Image.open(self.test_image_path)
        draw_image = draw_pose_on_image(pose_parts['bodies'], image, color=(255, 0, 0))
        draw_image = draw_pose_on_image(pose_parts['hands'], image, color=(0, 255, 0))
        draw_image = draw_pose_on_image(pose_parts['faces'], image, color=(0, 0, 255))
        draw_image = draw_pose_on_image(pose_parts['foot'], image, color=(0, 255, 255))
        draw_image.save('data/example_wholebody_image_draw_pose_parts.png')

if __name__ == '__main__':
    unittest.main()
