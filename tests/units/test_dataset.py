import unittest
from unittest import mock

import numpy as np
import pose3d_utils
import torch

from experimenting.dataset.dataset import (
    AutoEncoderDataset,
    ClassificationDataset,
    Joints3DDataset,
    Joints3DStereoDataset,
)

TEST_IMAGE_SHAPE = (224, 224)


class TestBaseDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is TestBaseDataset:
            raise unittest.SkipTest("Skip TestCore tests")
        super(TestBaseDataset, cls).setUpClass()

    def _get_mocked_skeleton(self):
        mocked_skeleton = mock.MagicMock()
        mocked_tensor = torch.rand((13, 3))
        mocked_skeleton._get_tensor.return_value = mocked_tensor
        mocked_skeleton.get_masked_skeleton.return_value = mocked_skeleton
        return mocked_skeleton, mocked_tensor

    def _get_skeletons(self):
        mocked_skeleton, mocked_skeleton_tensor = self._get_mocked_skeleton()
        (
            mocked_skeleton_projected,
            mocked_skeleton_projected_tensor,
        ) = self._get_mocked_skeleton()

        (
            mocked_skeleton_normalized,
            mocked_skeleton_normalized_tensor,
        ) = self._get_mocked_skeleton()

        mocked_skeleton.project_onto_camera.return_value = mocked_skeleton_projected

        mocked_skeleton_projected.normalize.return_value = mocked_skeleton_normalized

        return (
            mocked_skeleton,
            mocked_skeleton_tensor,
            mocked_skeleton_normalized_tensor,
            mocked_skeleton_projected_tensor,
        )

    def setUp(self):
        self.mocked_dataset = mock.MagicMock(N_JOINTS=13, in_shape=(224, 224))
        self.mocked_dataset.get_frame_from_id.return_value = "image"

        self.mocked_dataset.get_label_from_id.return_value = "label"
        (
            self.mocked_skeleton,
            self.mocked_skeleton_tensor,
            self.mocked_skeleton_normalized_tensor,
            self.mocked_skeleton_projected_tensor,
        ) = self._get_skeletons()

        self.mocked_joint = (
            self.mocked_skeleton,
            torch.rand((3, 4)),
            torch.rand((3, 4)),
        )

        self.mocked_dataset.get_joint_from_id.return_value = self.mocked_joint

        self.mocked_indexes = mock.MagicMock()
        self.mocked_indexes.__len__.return_value = 10
        self.mocked_indexes.__getitem__.side_effect = range(10, 19)
        self.mocked_transform = mock.MagicMock()
        self.mocked_transform.return_value = {'image': 'aug_image'}


class TestClassificationDataset(TestBaseDataset):
    def test_getitem_frame(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
        }
        task_dataset = ClassificationDataset(**dataset_config)
        idx = 0

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_label_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertEqual(y, "label")

    def test_getitem_frame_augmented(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
            'transform': self.mocked_transform,
        }
        task_dataset = ClassificationDataset(**dataset_config)
        idx = 0

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_label_from_id.assert_called_once()
        self.mocked_transform.assert_called_once_with(image="image")
        self.assertEqual(x, "aug_image")
        self.assertEqual(y, "label")


class TestAutoencoderDataset(TestBaseDataset):
    def test_getitem(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
        }
        task_dataset = AutoEncoderDataset(**dataset_config)
        idx = 0

        x = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.assertEqual(x, "image")

    def test_getitem_augmented(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
            'transform': self.mocked_transform,
        }
        task_dataset = AutoEncoderDataset(**dataset_config)
        idx = 0

        x = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_transform.assert_called_once_with(image="image")
        self.assertEqual(x, "aug_image")


class TestJoints3DDataset(TestBaseDataset):
    def test_getitem(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
        }

        task_dataset = Joints3DDataset(**dataset_config)
        mocked_normalizer = mock.Mock()
        task_dataset.normalizer = mocked_normalizer
        idx = 0
        expected_camera = self.mocked_joint[1]
        expected_M = self.mocked_joint[2]

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_joint_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertTrue(torch.equal(y['camera'], expected_camera))
        self.assertTrue(torch.equal(y['M'], expected_M))
        self.assertTrue(torch.equal(self.mocked_skeleton_tensor, y['xyz']))

        self.assertTrue(
            torch.equal(
                y['normalized_skeleton'], self.mocked_skeleton_normalized_tensor
            )
        )

        self.mocked_skeleton.project_onto_camera.assert_called_once()


class TestJoints3DStereoDataset(TestBaseDataset):
    def test_getitem(self):
        self.mocked_indexes.__getitem__.side_effect = np.concatenate(
            [np.expand_dims(np.arange(0, 10), 1), np.expand_dims(np.arange(11, 21), 1)],
            1,
        )
        self.mocked_dataset.get_frame_from_id.return_value = np.random.rand(224, 224, 3)
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
        }

        task_dataset = Joints3DStereoDataset(**dataset_config)
        mocked_normalizer = mock.Mock()
        task_dataset.normalizer = mocked_normalizer
        idx = 0
        expected_camera = self.mocked_joint[1]
        expected_M = self.mocked_joint[2]

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called()
        self.mocked_dataset.get_frame_from_id.assert_called()
        self.mocked_dataset.get_joint_from_id.assert_called_once()
        self.assertEqual(len(x), 2)
        self.assertTrue(torch.equal(y['camera'], expected_camera))
        self.assertTrue(torch.equal(y['M'], expected_M))
        self.assertTrue(torch.equal(self.mocked_skeleton_tensor, y['xyz']))

        self.assertTrue(
            torch.equal(
                y['normalized_skeleton'], self.mocked_skeleton_normalized_tensor
            )
        )

        self.mocked_skeleton.project_onto_camera.assert_called_once()
