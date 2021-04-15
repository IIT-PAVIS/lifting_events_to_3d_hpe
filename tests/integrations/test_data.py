import unittest
from unittest import mock

import numpy as np
from omegaconf import DictConfig

from experimenting.dataset import factory
from experimenting.dataset.core import DHP19Core, HumanCore, NTUCore


class TestDHP19For3DHPE(unittest.TestCase):
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
        self.factory = factory.Joints3DConstructor()
        self.factory.set_dataset_core(self.core)

    def test_split(self):
        train, val, test = self.factory.get_train_test_split()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)


class TestH3MFor3DHPE(unittest.TestCase):
    @mock.patch(
        "experimenting.dataset.core.h3mcore.HumanCore.get_pose_data", mock.Mock()
    )
    def setUp(self):
        data_dir = 'tests/data/h3m/'
        joints_path = 'tests/data/h3m/data_3d_h36m.npz'
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
        self.factory = factory.Joints3DConstructor()
        self.factory.set_dataset_core(self.core)

    def test_split(self):
        train, val, test = self.factory.get_train_test_split()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)


class TestH3MForStereo3DHPE(unittest.TestCase):
    @mock.patch(
        "experimenting.dataset.core.h3mcore.HumanCore.get_pose_data", mock.Mock()
    )
    def setUp(self):
        data_dir = 'tests/data/h3m/'
        joints_path = 'tests/data/h3m/data_3d_h36m.npz'
        self.hparams = DictConfig(
            {
                'name': 'test',
                'data_dir': data_dir,
                'joints_path': joints_path,
                'n_channels': 1,
                'partition': 'cross-subject',
                'test_subjects': [1],
            }
        )

        self.core = HumanCore(**self.hparams)
        self.factory = factory.Joints3DStereoConstructor()
        self.factory.set_dataset_core(self.core)

    def test_split(self):
        train, val, test = self.factory.get_train_test_split()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
