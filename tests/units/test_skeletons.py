import unittest

import torch

from experimenting import utils


class TestSkeleton(unittest.TestCase):
    precision_error = 1e-3  # acceptable precision error in mm for algebra calculation

    def setUp(self):
        self.joints = torch.randn(13, 3)
        self.sk = utils.skeleton_helpers.Skeleton(self.joints)

    def test_init(self):
        self.assertIsInstance(self.sk, utils.skeleton_helpers.Skeleton)
        self.assertTrue(torch.equal(self.sk._get_tensor(), self.joints))

    def test_torso_length(self):
        neck_point = self.joints.index_select(0, torch.LongTensor([1, 2])).mean(0)

        pelvic_point = self.joints.index_select(0, torch.LongTensor([5, 6])).mean(0)
        distance = torch.norm(neck_point - pelvic_point)
        self.assertEqual(self.sk.get_skeleton_torso_length(), distance)

    def test_normalize_denormalize_z_ref(self):
        camera = torch.randn(3, 4)
        height = 100
        width = 100
        normalized_skeleton = self.sk.normalize(height, width, camera)
        z_ref = normalized_skeleton.get_z_ref()
        denormalized_skeleton = normalized_skeleton.denormalize(
            height, width, camera, z_ref=z_ref
        )

        self.assertIsInstance(normalized_skeleton, utils.skeleton_helpers.Skeleton)
        self.assertIsInstance(denormalized_skeleton, utils.skeleton_helpers.Skeleton)

    def test_normalize_denormalize_torso(self):
        camera = torch.randn(3, 4)
        height = 100
        width = 100

        torso_length = self.sk.get_skeleton_torso_length()

        normalized_skeleton = self.sk.normalize(height, width, camera)
        denormalized_skeleton = normalized_skeleton.denormalize(
            height, width, camera, torso_length=torso_length
        )

        self.assertIsInstance(normalized_skeleton, utils.skeleton_helpers.Skeleton)
        self.assertIsInstance(denormalized_skeleton, utils.skeleton_helpers.Skeleton)
