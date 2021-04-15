import unittest
from unittest import mock

import numpy as np
from omegaconf import DictConfig

from experimenting.dataset.core import DHP19Core, HumanCore, NTUCore


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is TestCore:
            raise unittest.SkipTest("Skip TestCore tests")
        super(TestCore, cls).setUpClass()

    def test_init(self):
        self.assertIsNotNone(self.core)

    def test_paths_params(self):

        self.assertIsNotNone(self.core.file_paths)
        self.assertGreater(len(self.core.file_paths), 0)


class TestDHP19ParamsDefaultPartition(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3],
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'partition': None,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestDHP19ParamsCrossSubject(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3],
                'partition': 'cross-subject',
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestDHP19ParamsCrossView(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [0, 1, 2, 3],
                'partition': 'cross-view',
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestNTUParams(TestCore):
    def setUp(self):
        data_dir = 'tests/data/ntu/frames'
        labels_dir = 'tests/data/ntu/labels'
        self.hparams = DictConfig(
            {
                'name': 'test',
                'data_dir': data_dir,
                'labels_dir': labels_dir,
                'test_subjects': [19],
                'partition': 'cross-subject',
            }
        )
        self.core = NTUCore(**self.hparams)


class TestHumanCore(TestCore):
    @mock.patch(
        "experimenting.dataset.core.h3mcore.HumanCore.get_pose_data", mock.Mock()
    )
    def setUp(self):
        data_dir = 'tests/data/h3m/'
        joints_path = 'tests/data/h3m/labels.npz'
        self.hparams = DictConfig(
            {
                'name': 'test',
                'data_dir': data_dir,
                'joints_path': joints_path,
                'n_channels': 1,
                'partition': 'cross-subject',
                'test_subjects': [5],
            }
        )

        self.core = HumanCore(**self.hparams)

    def test_get_frame_info(self):
        file_path = 'tests/data/h3m/S1/Directions 1.54138969S1/frame0000001.npy'
        expected_info = {
            'cam': 0,
            'action': 'Directions 1',
            'subject': 1,
            'frame': '0000001',
        }

        result_info = HumanCore.get_frame_info(file_path)

        self.assertEqual(expected_info, result_info)

    def test_get_frame_info_noisy(self):
        file_path = 'tests/data/h3m/S1/Videos/Directions 1.55011271/frame0000001.npy'
        expected_info = {
            'cam': 1,
            'action': 'Directions 1',
            'subject': 1,
            'frame': '0000001',
        }

        result_info = HumanCore.get_frame_info(file_path)

        self.assertEqual(expected_info, result_info)

    def test_get_frame_info_different_values(self):
        file_path = 'tests/data/h3m/S1/Videos/Directions.60457274/frame0000010.npy'
        expected_info = {
            'cam': 3,
            'action': 'Directions',
            'subject': 1,
            'frame': '0000010',
        }

        result_info = HumanCore.get_frame_info(file_path)

        self.assertEqual(expected_info, result_info)

    def test_get_label(self):
        file_path = 'tests/data/h3m/S1/Directions 1.54138969S1/frame0000001.npy'

        expected = 0

        self.assertEqual(HumanCore.get_label_from_filename(file_path), expected)

    @mock.patch("numpy.load")
    def test_get_labels(self, numpy_load_mocked):
        mocked_data_poses = mock.Mock()
        mocked_data_timestamps = mock.Mock()
        n_frames = 10
        n_joints = 17
        sub_n = 1
        mocked_poses = np.random.rand(n_frames, n_joints, 3)
        mocked_timestamps = np.random.rand(n_frames)
        mocked_data_poses.item.return_value = {f'S{sub_n}': {'Directions': mocked_poses}}
        mocked_data_timestamps.item.return_value = {f'S{sub_n}': {'Directions': mocked_timestamps}}
        mocked_pose_loaded = {'positions_3d': mocked_data_poses, 'timestamps': mocked_data_timestamps}
        numpy_load_mocked.return_value = mocked_pose_loaded
        path = "path/to/data.npz"

        data = HumanCore.get_pose_data(path)

        self.assertIsNotNone(data)
        self.assertTrue(sub_n in data)
        self.assertTrue("Directions" in data[sub_n])
        self.assertTrue("positions" in data[sub_n]["Directions"])

        self.assertTrue("extrinsics" in data[sub_n]["Directions"])


if __name__ == '__main__':
    unittest.main()
