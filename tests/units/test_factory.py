import unittest
from unittest import mock

import numpy as np
from omegaconf import DictConfig

from experimenting.dataset.core import DHP19Core, NTUCore
from experimenting.dataset.factory import (
    AutoEncoderConstructor,
    ClassificationConstructor,
    HeatmapConstructor,
    Joints3DConstructor,
)


class TestFactoryDHP19(unittest.TestCase):
    def setUp(self):
        self.core = mock.MagicMock(in_shape=(224, 224))

        self.indexes = [1, 2, 3]
        self.params = {
            'indexes': self.indexes,
            'augmentation_config': {'apply': {}},
        }

    def test_ae(self):
        data_constructor = AutoEncoderConstructor()
        self.assertIsNotNone(data_constructor)
        train = data_constructor.get_dataset(**self.params)

        self.assertGreater(len(train), 0)

    def test_classification(self):
        data_constructor = ClassificationConstructor()
        self.assertIsNotNone(data_constructor)
        train = data_constructor.get_dataset(**self.params)

        self.assertGreater(len(train), 0)

    def test_3d_joints(self):
        data_constructor = Joints3DConstructor()
        data_constructor.set_dataset_core(self.core)

        self.assertIsNotNone(data_constructor)
        train = data_constructor.get_dataset(**self.params)
        self.assertGreater(len(train), 0)

    def test_hm(self):

        data_constructor = HeatmapConstructor()
        self.assertIsNotNone(data_constructor)

        train = data_constructor.get_dataset(**self.params)
        self.assertGreater(len(train), 0)


if __name__ == '__main__':
    unittest.main()
